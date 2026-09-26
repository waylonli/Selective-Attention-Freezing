"""Isolated-process nanoGPT prefill benchmark for frozen-head Triton attention.

This measures a full, target-free ``model(x)`` forward.  It is a prefill (square
causal attention) benchmark, not a KV-cache decode benchmark.  The reference is
forced onto PyTorch's Flash SDPA backend; each frozen-rate variant is verified
to call ``fused_mixed_attn``.

Every (context, variant, repeat) runs in a fresh spawned process.  Consequently
only one model is resident at a time, allocator peaks do not leak across
variants, and rebuilding with the same seed gives matching model weights.
Results are emitted as JSON Lines on stdout.

Example:
    python nanogpt/bench_inference_triton.py \
        --T 1024,2048,4096 --B 4 --rates 0.25,0.5,0.75 \
        --warmup 10 --iters 30 --repeats 3

For order-controlled comparisons, add ``--paired-abba`` and use at least two
repeats.  Repeat 0 runs Flash then fused (AB), repeat 1 runs fused then Flash
(BA), and so on; every leg still gets a fresh process.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

# model.py reads this at import time.  This benchmark is specifically for the
# fused implementation, so do not silently inherit a split-path setting.
os.environ["FROZEN_ATTN_FUSED"] = "1"

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model as model_mod  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


VOCAB_SIZE = 50304
GIB = 1024**3


@dataclass(frozen=True)
class WorkerSpec:
    T: int
    B: int
    n_layer: int
    n_head: int
    n_embd: int
    rate: float
    dynamic: bool
    warmup: int
    iters: int
    repeat: int
    seed: int
    vocab_size: int
    freeze_pattern_dtype: str
    prior_repr: str


def _attn_modules(model: GPT) -> list[Any]:
    modules = []
    for block in model.transformer.h:
        attn = block.main_block
        if hasattr(attn, "block"):
            attn = attn.block
        modules.append(attn)
    return modules


def _set_freeze(model: GPT, requested_rate: float, T: int) -> float:
    """Freeze the same rounded number of heads in every layer."""
    modules = _attn_modules(model)
    k = int(round(requested_rate * model.config.n_head))
    k = min(max(k, 0), model.config.n_head)

    if k and model.config.rand_attn_prior_repr == "decomposed":
        # alpha=rho=0 is exactly the same causal-uniform prior used below:
        # softmax(alpha[j] + rho[i-j]) = 1 / (i+1), j <= i.
        zeros = torch.zeros((k, T), device="cuda", dtype=torch.float32)
        for attn in modules:
            attn.freeze_heads_decomposed(range(k), zeros, zeros)
        del zeros
    elif k:
        causal = torch.tril(
            torch.ones((T, T), device="cuda", dtype=torch.bfloat16)
        )
        pattern = causal / causal.sum(dim=-1, keepdim=True)
        patterns = pattern.unsqueeze(0).expand(k, -1, -1)
        for attn in modules:
            attn.freeze_heads(range(k), patterns)
        del patterns, pattern, causal

    frozen = sum(int(attn.dyn_frozen.sum().item()) for attn in modules)
    total = len(modules) * model.config.n_head
    return frozen / total


def _pattern_bytes(model: GPT) -> int:
    """Bytes held by the fixed-prior representation (dense or decomposed)."""
    total = 0
    for attn in _attn_modules(model):
        names = ("dyn_pattern",) if attn.prior_repr == "dense" else (
            "dyn_pattern", "dyn_alpha", "dyn_rho", "dyn_z", "dyn_rho_band"
        )
        for name in names:
            value = getattr(attn, name, None)
            if value is not None:
                total += int(value.numel() * value.element_size())
    return total


def _gib(n: int | float) -> float:
    return float(n) / GIB


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _run_worker(spec: WorkerSpec) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if spec.n_embd % spec.n_head:
        raise ValueError("n_embd must be divisible by n_head")

    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(
        n_layer=spec.n_layer,
        n_head=spec.n_head,
        n_embd=spec.n_embd,
        block_size=spec.T,
        vocab_size=spec.vocab_size,
        dropout=0.0,
        bias=False,
        rand_attn_dynamic=spec.dynamic,
        rand_seed=spec.seed,
        freeze_pattern_dtype=spec.freeze_pattern_dtype,
        rand_attn_prior_repr=spec.prior_repr,
    )
    model = GPT(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    parameter_count = sum(p.numel() for p in model.parameters())

    generator = torch.Generator(device="cuda").manual_seed(spec.seed + 1)
    x = torch.randint(
        0, spec.vocab_size, (spec.B, spec.T), device="cuda", generator=generator
    )

    if spec.dynamic:
        if not model_mod._FUSED:
            raise RuntimeError("FROZEN_ATTN_FUSED is disabled")
        actual_rate = _set_freeze(model, spec.rate, spec.T)
    else:
        actual_rate = 0.0

    pattern_bytes = _pattern_bytes(model)
    fused_calls = 0

    # Create Python/CUDA bookkeeping before warming the GPU. In the previous
    # order, gc.collect() plus hundreds of Event allocations happened after the
    # warmup and let GH200 clocks fall back down; early measured samples were
    # then systematically slower and biased whichever variant ran first.
    gc.collect()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(spec.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(spec.iters)]

    def step() -> None:
        # No targets: this deliberately excludes full-vocabulary logits at every
        # token and matches autoregressive prefill's target-free model forward.
        model(x)

    # Force Flash for every SDPA call.  The baseline therefore fails loudly if
    # Flash cannot serve the requested shape; the dynamic fused call is unaffected.
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        sdpa_kernel([SDPBackend.FLASH_ATTENTION]),
    ):
        if spec.dynamic and actual_rate > 0:
            original_fused = model_mod.fused_mixed_attn

            def counted_fused(*args: Any, **kwargs: Any) -> Any:
                nonlocal fused_calls
                fused_calls += 1
                return original_fused(*args, **kwargs)

            model_mod.fused_mixed_attn = counted_fused
            try:
                step()  # also pays Triton compilation before timing
            finally:
                model_mod.fused_mixed_attn = original_fused
            if fused_calls != spec.n_layer:
                raise RuntimeError(
                    "fused kernel dispatch verification failed: "
                    f"expected {spec.n_layer} calls, observed {fused_calls}"
                )
        else:
            # Validates that the reference/no-freeze SDPA path supports Flash.
            step()

        for _ in range(spec.warmup):
            step()

        torch.cuda.synchronize()

        # Stable tensors after compilation/warmup: weights, input, patterns and
        # inference weight caches.  This is the deployment-resident allocation.
        resident_allocated = int(torch.cuda.memory_allocated())
        resident_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()

        for start, end in zip(starts, ends):
            start.record()
            step()
            end.record()
        torch.cuda.synchronize()

    samples_ms = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    incremental_peak = max(0, peak_allocated - resident_allocated)
    median_ms = float(statistics.median(samples_ms))

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "type": "run",
        "phase": "prefill",
        "variant": "fused_frozen" if spec.dynamic else "flash_sdpa",
        "T": spec.T,
        "B": spec.B,
        "n_layer": spec.n_layer,
        "n_head": spec.n_head,
        "n_embd": spec.n_embd,
        "head_dim": spec.n_embd // spec.n_head,
        "parameter_count": parameter_count,
        "requested_rate": spec.rate if spec.dynamic else 0.0,
        "actual_rate": actual_rate,
        "frozen_heads_per_layer": int(round(actual_rate * spec.n_head)),
        "repeat": spec.repeat,
        "seed": spec.seed,
        "warmup": spec.warmup,
        "iters": spec.iters,
        "median_ms": median_ms,
        "mean_ms": float(statistics.fmean(samples_ms)),
        "p10_ms": _quantile(samples_ms, 0.10),
        "p90_ms": _quantile(samples_ms, 0.90),
        "tokens_per_second": spec.B * spec.T / (median_ms / 1000.0),
        "samples_ms": samples_ms,
        "resident_allocated_bytes": resident_allocated,
        "resident_allocated_gib": _gib(resident_allocated),
        "resident_reserved_bytes": resident_reserved,
        "resident_reserved_gib": _gib(resident_reserved),
        "pattern_bytes": pattern_bytes,
        "pattern_gib": _gib(pattern_bytes),
        "incremental_peak_bytes": incremental_peak,
        "incremental_peak_gib": _gib(incremental_peak),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gib": _gib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": _gib(peak_reserved),
        "flash_sdpa_forced": True,
        "fused_kernel_verified": bool(spec.dynamic and actual_rate > 0),
        "fused_calls_in_validation": fused_calls,
        "isolation": "spawn_process_per_variant_repeat",
        "dtype": "bfloat16",
        "freeze_pattern_dtype": spec.freeze_pattern_dtype,
        "prior_repr": spec.prior_repr,
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _worker_entry(spec_dict: dict[str, Any], send: Any) -> None:
    # Keep worker/model diagnostics out of the JSONL stream.  If a run fails,
    # return the captured tail with the traceback for a useful parent-side error.
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = _run_worker(WorkerSpec(**spec_dict))
        send.send({"ok": True, "result": result})
    except BaseException:
        send.send(
            {
                "ok": False,
                "traceback": traceback.format_exc(),
                "worker_log_tail": (stdout.getvalue() + stderr.getvalue())[-8000:],
            }
        )
    finally:
        send.close()


def _spawn_one(ctx: Any, spec: WorkerSpec) -> dict[str, Any]:
    recv, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker_entry, args=(asdict(spec), send))
    process.start()
    send.close()
    try:
        message = recv.recv()
    except EOFError as exc:
        process.join()
        raise RuntimeError(
            f"worker exited without a result (exit code {process.exitcode})"
        ) from exc
    finally:
        recv.close()
    process.join()
    if not message["ok"]:
        raise RuntimeError(
            "benchmark worker failed:\n"
            + message["traceback"]
            + "\nworker output tail:\n"
            + message["worker_log_tail"]
        )
    if process.exitcode != 0:
        raise RuntimeError(f"worker exited with code {process.exitcode}")
    return message["result"]


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _summary(rows: list[dict[str, Any]], baseline_ms: float) -> dict[str, Any]:
    medians = [float(row["median_ms"]) for row in rows]
    resident = [int(row["resident_allocated_bytes"]) for row in rows]
    pattern = [int(row["pattern_bytes"]) for row in rows]
    incremental = [int(row["incremental_peak_bytes"]) for row in rows]
    median_ms = float(statistics.median(medians))
    first = rows[0]
    return {
        "type": "summary",
        "phase": "prefill",
        "variant": first["variant"],
        "T": first["T"],
        "B": first["B"],
        "n_layer": first["n_layer"],
        "n_head": first["n_head"],
        "n_embd": first["n_embd"],
        "requested_rate": first["requested_rate"],
        "actual_rate": first["actual_rate"],
        "prior_repr": first["prior_repr"],
        "repeats": len(rows),
        "median_ms": median_ms,
        "min_repeat_ms": min(medians),
        "max_repeat_ms": max(medians),
        "tokens_per_second": first["B"] * first["T"] / (median_ms / 1000.0),
        "speedup_vs_flash": baseline_ms / median_ms,
        "resident_allocated_bytes": int(statistics.median(resident)),
        "resident_allocated_gib": _gib(statistics.median(resident)),
        "pattern_bytes": int(statistics.median(pattern)),
        "pattern_gib": _gib(statistics.median(pattern)),
        "incremental_peak_bytes": int(statistics.median(incremental)),
        "incremental_peak_gib": _gib(statistics.median(incremental)),
        "flash_sdpa_forced": True,
        "fused_kernel_verified": all(row["fused_kernel_verified"] for row in rows)
        if first["actual_rate"] > 0
        else False,
        "isolation": "spawn_process_per_variant_repeat",
    }


def _paired_run(
    baseline: dict[str, Any],
    fused: dict[str, Any],
    order: str,
) -> dict[str, Any]:
    """One order-controlled Flash/fused comparison from two fresh workers."""
    baseline_ms = float(baseline["median_ms"])
    fused_ms = float(fused["median_ms"])
    return {
        "type": "paired_run",
        "phase": "prefill",
        "mode": "paired_abba",
        "T": fused["T"],
        "B": fused["B"],
        "n_layer": fused["n_layer"],
        "n_head": fused["n_head"],
        "n_embd": fused["n_embd"],
        "requested_rate": fused["requested_rate"],
        "actual_rate": fused["actual_rate"],
        "prior_repr": fused["prior_repr"],
        "repeat": fused["repeat"],
        "order": order,
        "execution_order": (
            ["flash_sdpa", "fused_frozen"]
            if order == "AB"
            else ["fused_frozen", "flash_sdpa"]
        ),
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "paired_delta_ms": baseline_ms - fused_ms,
        "paired_speedup": baseline_ms / fused_ms,
        "baseline_tokens_per_second": baseline["tokens_per_second"],
        "fused_tokens_per_second": fused["tokens_per_second"],
        "baseline_resident_allocated_bytes": baseline[
            "resident_allocated_bytes"
        ],
        "fused_resident_allocated_bytes": fused["resident_allocated_bytes"],
        "fused_pattern_bytes": fused["pattern_bytes"],
        "freeze_pattern_dtype": fused["freeze_pattern_dtype"],
        "baseline_incremental_peak_bytes": baseline["incremental_peak_bytes"],
        "fused_incremental_peak_bytes": fused["incremental_peak_bytes"],
        "flash_sdpa_forced": baseline["flash_sdpa_forced"],
        "fused_kernel_verified": fused["fused_kernel_verified"],
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def _paired_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Robust aggregate of within-pair ratios, with order effects exposed."""
    first = pairs[0]
    speedups = [float(pair["paired_speedup"]) for pair in pairs]
    baseline_ms = [float(pair["baseline_ms"]) for pair in pairs]
    fused_ms = [float(pair["fused_ms"]) for pair in pairs]
    ab = [float(pair["paired_speedup"]) for pair in pairs if pair["order"] == "AB"]
    ba = [float(pair["paired_speedup"]) for pair in pairs if pair["order"] == "BA"]
    baseline_resident = [
        int(pair["baseline_resident_allocated_bytes"]) for pair in pairs
    ]
    fused_resident = [int(pair["fused_resident_allocated_bytes"]) for pair in pairs]
    fused_pattern = [int(pair["fused_pattern_bytes"]) for pair in pairs]
    baseline_incremental = [
        int(pair["baseline_incremental_peak_bytes"]) for pair in pairs
    ]
    fused_incremental = [int(pair["fused_incremental_peak_bytes"]) for pair in pairs]
    median_speedup = float(statistics.median(speedups))
    median_baseline_ms = float(statistics.median(baseline_ms))
    median_fused_ms = float(statistics.median(fused_ms))
    return {
        "type": "paired_summary",
        "phase": "prefill",
        "mode": "paired_abba",
        "T": first["T"],
        "B": first["B"],
        "n_layer": first["n_layer"],
        "n_head": first["n_head"],
        "n_embd": first["n_embd"],
        "requested_rate": first["requested_rate"],
        "actual_rate": first["actual_rate"],
        "prior_repr": first["prior_repr"],
        "pairs": len(pairs),
        "ab_pairs": len(ab),
        "ba_pairs": len(ba),
        "order_balanced": abs(len(ab) - len(ba)) <= 1,
        "paired_estimator": "median_of_within_pair_speedup_ratios",
        "paired_speedup_median": median_speedup,
        "paired_speedup_min": min(speedups),
        "paired_speedup_max": max(speedups),
        "paired_speedup_p25": _quantile(speedups, 0.25),
        "paired_speedup_p75": _quantile(speedups, 0.75),
        "ab_speedup_median": float(statistics.median(ab)) if ab else None,
        "ba_speedup_median": float(statistics.median(ba)) if ba else None,
        # Keep this familiar field name available for downstream result readers,
        # but define it as the paired median rather than a ratio of pooled runs.
        "speedup_vs_flash": median_speedup,
        "baseline_median_ms": median_baseline_ms,
        "fused_median_ms": median_fused_ms,
        "ratio_of_median_times": median_baseline_ms / median_fused_ms,
        "baseline_resident_allocated_bytes": int(
            statistics.median(baseline_resident)
        ),
        "fused_resident_allocated_bytes": int(statistics.median(fused_resident)),
        "fused_pattern_bytes": int(statistics.median(fused_pattern)),
        "freeze_pattern_dtype": first["freeze_pattern_dtype"],
        "baseline_incremental_peak_bytes": int(
            statistics.median(baseline_incremental)
        ),
        "fused_incremental_peak_bytes": int(statistics.median(fused_incremental)),
        "flash_sdpa_forced": all(pair["flash_sdpa_forced"] for pair in pairs),
        "fused_kernel_verified": all(
            pair["fused_kernel_verified"] for pair in pairs
        )
        if first["actual_rate"] > 0
        else False,
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def _run_paired_context(
    ctx: Any,
    args: argparse.Namespace,
    T: int,
    rates: list[float],
) -> None:
    """Run AB, BA, AB, ... pairs independently for every requested rate."""
    for rate in rates:
        pairs = []
        for repeat in range(args.repeats):
            baseline_spec = WorkerSpec(
                T=T,
                B=args.B,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                rate=0.0,
                dynamic=False,
                warmup=args.warmup,
                iters=args.iters,
                repeat=repeat,
                seed=args.seed,
                vocab_size=args.vocab_size,
                freeze_pattern_dtype=args.freeze_pattern_dtype,
                prior_repr=args.prior_repr,
            )
            fused_spec = WorkerSpec(
                T=T,
                B=args.B,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                rate=rate,
                dynamic=True,
                warmup=args.warmup,
                iters=args.iters,
                repeat=repeat,
                seed=args.seed,
                vocab_size=args.vocab_size,
                freeze_pattern_dtype=args.freeze_pattern_dtype,
                prior_repr=args.prior_repr,
            )
            order = "AB" if repeat % 2 == 0 else "BA"
            ordered = (
                (("baseline", baseline_spec), ("fused", fused_spec))
                if order == "AB"
                else (("fused", fused_spec), ("baseline", baseline_spec))
            )
            results = {}
            for label, spec in ordered:
                row = _spawn_one(ctx, spec)
                results[label] = row
                # Preserve the existing per-worker run schema in paired mode.
                print(json.dumps(row, sort_keys=True), flush=True)
            pair = _paired_run(results["baseline"], results["fused"], order)
            pairs.append(pair)
            print(json.dumps(pair, sort_keys=True), flush=True)
        print(json.dumps(_paired_summary(pairs), sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--T",
        default="1024",
        help="context length, or comma-separated context lengths",
    )
    parser.add_argument("--B", "--batch", dest="B", type=int, default=8)
    parser.add_argument("--n-layer", "--n_layer", dest="n_layer", type=int, default=12)
    parser.add_argument("--n-head", "--n_head", dest="n_head", type=int, default=12)
    parser.add_argument("--n-embd", "--n_embd", dest="n_embd", type=int, default=768)
    parser.add_argument("--rates", default="0.25,0.5,0.75")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--paired-abba",
        "--paired",
        dest="paired_abba",
        action="store_true",
        help=(
            "pair each fused run with a fresh Flash run and alternate AB/BA "
            "execution order across repeats (requires repeats >= 2)"
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument(
        "--freeze-pattern-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="dtype of the resident fixed-pattern buffers",
    )
    parser.add_argument(
        "--prior-repr",
        choices=("dense", "decomposed"),
        default="dense",
        help="fixed-prior storage used by the fused arm",
    )
    args = parser.parse_args()

    contexts = _csv_ints(args.T)
    rates = _csv_floats(args.rates)
    if not contexts or any(T <= 0 for T in contexts):
        parser.error("--T must contain positive integers")
    if not rates or any(rate < 0 or rate > 1 for rate in rates):
        parser.error("--rates must contain values in [0, 1]")
    if args.B <= 0 or args.n_layer <= 0 or args.n_head <= 0 or args.n_embd <= 0:
        parser.error("model dimensions and batch size must be positive")
    if args.n_embd % args.n_head:
        parser.error("--n-embd must be divisible by --n-head")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be >= 0; iters and repeats must be > 0")
    if args.paired_abba and args.repeats < 2:
        parser.error("--paired-abba requires --repeats >= 2 to include AB and BA")

    # Spawn is mandatory: fork after importing torch is unsafe for CUDA, and a
    # new process is what makes each model/allocator measurement independent.
    ctx = mp.get_context("spawn")
    try:
        for T in contexts:
            if args.paired_abba:
                _run_paired_context(ctx, args, T, rates)
                continue

            baseline_rows = []
            for repeat in range(args.repeats):
                spec = WorkerSpec(
                    T=T,
                    B=args.B,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    rate=0.0,
                    dynamic=False,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeat=repeat,
                    seed=args.seed,
                    vocab_size=args.vocab_size,
                    freeze_pattern_dtype=args.freeze_pattern_dtype,
                    prior_repr=args.prior_repr,
                )
                row = _spawn_one(ctx, spec)
                baseline_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

            baseline_ms = float(
                statistics.median(row["median_ms"] for row in baseline_rows)
            )
            print(
                json.dumps(_summary(baseline_rows, baseline_ms), sort_keys=True),
                flush=True,
            )

            for rate in rates:
                rows = []
                for repeat in range(args.repeats):
                    spec = WorkerSpec(
                        T=T,
                        B=args.B,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        n_embd=args.n_embd,
                        rate=rate,
                        dynamic=True,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeat=repeat,
                        seed=args.seed,
                        vocab_size=args.vocab_size,
                        freeze_pattern_dtype=args.freeze_pattern_dtype,
                        prior_repr=args.prior_repr,
                    )
                    row = _spawn_one(ctx, spec)
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                print(json.dumps(_summary(rows, baseline_ms), sort_keys=True), flush=True)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "type": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()

"""ABBA full-optimizer-update timing for a real full/frozen checkpoint pair.

Each timed sample mirrors one nanoGPT optimizer update: a fixed number of
micro-batches, BF16-autocast forward and cross-entropy, scaled backward,
gradient clipping, fused AdamW, and ``zero_grad(set_to_none=True)``.  The total
tokens per update are held fixed while micro-batch size changes.  Every ABBA
leg runs in a fresh spawned process and loads the learned head mask and prior
from a model-only checkpoint.

This benchmark measures model/optimizer compute.  It deliberately excludes
data loading and the one-time head-selection/prior-fitting intervention.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any

os.environ["FROZEN_ATTN_FUSED"] = "1"

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as model_mod  # noqa: E402
from bench_inference_triton import (  # noqa: E402
    _attn_modules,
    _gib,
    _pattern_bytes,
    _quantile,
)
from eval_nanogpt_logprobs import load_model  # noqa: E402


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    torch.set_float32_matmul_precision("high")

    model, cfg, checkpoint = load_model(spec["ckpt"], "cuda")
    del checkpoint
    if spec["T"] > cfg.block_size:
        raise ValueError(
            f"T={spec['T']} exceeds checkpoint block_size={cfg.block_size}")
    model.train()
    modules = _attn_modules(model)
    frozen_heads = sum(int(module.dyn_frozen.sum()) for module in modules)
    total_heads = len(modules) * cfg.n_head
    actual_rate = frozen_heads / total_heads
    fused_layers = sum(int(module.dyn_frozen.sum()) > 0 for module in modules)
    if spec["variant"] == "baseline" and frozen_heads:
        raise RuntimeError(
            f"baseline checkpoint contains {frozen_heads} frozen heads")
    if spec["variant"] == "frozen" and not frozen_heads:
        raise RuntimeError("frozen checkpoint contains no frozen heads")
    if spec["variant"] == "frozen" and not model_mod._FUSED:
        raise RuntimeError("FROZEN_ATTN_FUSED is disabled")

    optimizer = model.configure_optimizers(
        weight_decay=spec["weight_decay"],
        learning_rate=spec["learning_rate"],
        betas=(spec["beta1"], spec["beta2"]),
        device_type="cuda",
    )
    optimizer.zero_grad(set_to_none=True)

    generator = torch.Generator(device="cuda").manual_seed(spec["seed"] + 1)
    # Shift a T+1 token window exactly as nanoGPT's data loader does. Keeping
    # every micro-batch resident removes storage latency from both timing arms.
    token_windows = torch.randint(
        0,
        cfg.vocab_size,
        (spec["grad_accum"], spec["B"], spec["T"] + 1),
        device="cuda",
        generator=generator,
    )
    # Materialize the shifted views once, outside the timed region. nanoGPT's
    # loss flattens targets with ``view`` and therefore expects a contiguous
    # tensor, as provided by its real data loader. Keeping both tensors
    # resident also ensures the benchmark excludes data preparation/copies.
    inputs = token_windows[..., :-1].contiguous()
    targets_all = token_windows[..., 1:].contiguous()
    del token_windows
    input_checksum = int(inputs.sum(dtype=torch.int64).cpu())
    target_checksum = int(targets_all.sum(dtype=torch.int64).cpu())
    tokens_per_update = spec["grad_accum"] * spec["B"] * spec["T"]
    if tokens_per_update != spec["tokens_per_update"]:
        raise RuntimeError(
            f"token invariant failed: {tokens_per_update} != "
            f"{spec['tokens_per_update']}")

    def optimizer_update() -> None:
        last_loss = None
        for micro in range(spec["grad_accum"]):
            x = inputs[micro]
            targets = targets_all[micro]
            logits, loss = model(x, targets)
            if loss is None:
                raise RuntimeError("training forward did not return a loss")
            (loss / spec["grad_accum"]).backward()
            last_loss = loss.detach()
            del logits, loss
        if spec["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if last_loss is None:
            raise RuntimeError("no micro-batches executed")

    fused_calls = 0
    validation_loss = float("nan")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    flash = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    with autocast, flash:
        # Verify the actual dispatch before timing. Layers with no selected
        # frozen heads correctly remain on ordinary Flash SDPA.
        if spec["variant"] == "frozen":
            original = model_mod.fused_mixed_attn

            def counted(*args: Any, **kwargs: Any) -> Any:
                nonlocal fused_calls
                fused_calls += 1
                return original(*args, **kwargs)

            model_mod.fused_mixed_attn = counted
            try:
                x = inputs[0]
                targets = targets_all[0]
                logits, loss = model(x, targets)
                validation_loss = float(loss.detach().float().cpu())
                del logits, loss
            finally:
                model_mod.fused_mixed_attn = original
            if fused_calls != fused_layers:
                raise RuntimeError(
                    "fused dispatch verification failed: "
                    f"expected {fused_layers} calls, observed {fused_calls}")
        else:
            x = inputs[0]
            targets = targets_all[0]
            logits, loss = model(x, targets)
            validation_loss = float(loss.detach().float().cpu())
            del logits, loss

        optimizer.zero_grad(set_to_none=True)
        for _ in range(spec["warmup"]):
            optimizer_update()

        torch.cuda.synchronize()
        resident_allocated = int(torch.cuda.memory_allocated())
        resident_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(spec["iters"])]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(spec["iters"])]
        for begin, end in zip(starts, ends):
            begin.record()
            optimizer_update()
            end.record()
        torch.cuda.synchronize()

    samples = [float(begin.elapsed_time(end)) for begin, end in zip(starts, ends)]
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    median_ms = float(statistics.median(samples))
    properties = torch.cuda.get_device_properties(0)
    return {
        "type": "run",
        "phase": "training_optimizer_update",
        "variant": "flash_sdpa" if spec["variant"] == "baseline" else "fused_frozen",
        "checkpoint": spec["ckpt"],
        "T": spec["T"],
        "B": spec["B"],
        "grad_accum": spec["grad_accum"],
        "tokens_per_update": tokens_per_update,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "frozen_heads": frozen_heads,
        "fused_layers": fused_layers,
        "actual_rate": actual_rate,
        "repeat": spec["repeat"],
        "seed": spec["seed"],
        "warmup": spec["warmup"],
        "iters": spec["iters"],
        "median_ms": median_ms,
        "mean_ms": float(statistics.fmean(samples)),
        "p10_ms": _quantile(samples, 0.10),
        "p90_ms": _quantile(samples, 0.90),
        "tokens_per_second": tokens_per_update / (median_ms / 1000.0),
        "samples_ms": samples,
        "validation_loss": validation_loss,
        "input_checksum": input_checksum,
        "target_checksum": target_checksum,
        "resident_allocated_bytes": resident_allocated,
        "resident_allocated_gib": _gib(resident_allocated),
        "resident_reserved_bytes": resident_reserved,
        "resident_reserved_gib": _gib(resident_reserved),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gib": _gib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": _gib(peak_reserved),
        "incremental_peak_bytes": max(0, peak_allocated - resident_allocated),
        "pattern_bytes": _pattern_bytes(model),
        "flash_sdpa_forced": True,
        "fused_kernel_verified": (
            spec["variant"] == "frozen" and fused_calls == fused_layers),
        "fused_calls_in_validation": fused_calls,
        "gradient_clip_included": True,
        "optimizer_step_included": True,
        "optimizer": "fused_adamw",
        "data_loading_included": False,
        "selector_fit_included": False,
        "step_definition": (
            "grad_accum*(autocast_forward_cross_entropy+scaled_backward)"
            "+clip_grad_norm+fused_adamw.step+zero_grad"),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "autocast_dtype": "bfloat16",
        "prior_repr": cfg.rand_attn_prior_repr,
        "freeze_pattern_dtype": cfg.freeze_pattern_dtype,
        "device": properties.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def _entry(spec: dict[str, Any], send: Any) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = _worker(spec)
        send.send({"ok": True, "result": result})
    except BaseException:
        send.send({
            "ok": False,
            "traceback": traceback.format_exc(),
            "log": (stdout.getvalue() + stderr.getvalue())[-12000:],
        })
    finally:
        send.close()


def _spawn(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    recv, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_entry, args=(spec, send))
    process.start()
    send.close()
    try:
        message = recv.recv()
    except EOFError as exc:
        process.join()
        raise RuntimeError(
            f"worker exited without a result (exit code {process.exitcode})") from exc
    finally:
        recv.close()
    process.join()
    if not message["ok"]:
        raise RuntimeError(message["traceback"] + "\nworker output:\n" + message["log"])
    if process.exitcode:
        raise RuntimeError(f"worker exit code {process.exitcode}")
    return message["result"]


def _paired_run(baseline: dict[str, Any], frozen: dict[str, Any], order: str) -> dict[str, Any]:
    for field in ("T", "B", "grad_accum", "tokens_per_update", "input_checksum", "target_checksum"):
        if baseline[field] != frozen[field]:
            raise RuntimeError(
                f"paired-arm mismatch for {field}: {baseline[field]} != {frozen[field]}")
    baseline_ms = float(baseline["median_ms"])
    frozen_ms = float(frozen["median_ms"])
    return {
        "type": "paired_run",
        "phase": "training_optimizer_update",
        "order": order,
        "T": frozen["T"],
        "B": frozen["B"],
        "grad_accum": frozen["grad_accum"],
        "tokens_per_update": frozen["tokens_per_update"],
        "actual_rate": frozen["actual_rate"],
        "repeat": frozen["repeat"],
        "baseline_ms": baseline_ms,
        "fused_ms": frozen_ms,
        "paired_delta_ms": baseline_ms - frozen_ms,
        "paired_speedup": baseline_ms / frozen_ms,
        "baseline_tokens_per_second": baseline["tokens_per_second"],
        "fused_tokens_per_second": frozen["tokens_per_second"],
        "baseline_resident_allocated_bytes": baseline["resident_allocated_bytes"],
        "fused_resident_allocated_bytes": frozen["resident_allocated_bytes"],
        "baseline_resident_reserved_bytes": baseline["resident_reserved_bytes"],
        "fused_resident_reserved_bytes": frozen["resident_reserved_bytes"],
        "baseline_peak_allocated_bytes": baseline["peak_allocated_bytes"],
        "fused_peak_allocated_bytes": frozen["peak_allocated_bytes"],
        "baseline_peak_reserved_bytes": baseline["peak_reserved_bytes"],
        "fused_peak_reserved_bytes": frozen["peak_reserved_bytes"],
        "fused_pattern_bytes": frozen["pattern_bytes"],
        "flash_sdpa_forced": baseline["flash_sdpa_forced"],
        "fused_kernel_verified": frozen["fused_kernel_verified"],
        "optimizer_step_included": True,
        "data_loading_included": False,
        "selector_fit_included": False,
        "input_checksum": frozen["input_checksum"],
        "target_checksum": frozen["target_checksum"],
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def _paired_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    first = pairs[0]
    speedups = [float(row["paired_speedup"]) for row in pairs]
    baseline_ms = [float(row["baseline_ms"]) for row in pairs]
    frozen_ms = [float(row["fused_ms"]) for row in pairs]
    ab = [value for value, row in zip(speedups, pairs) if row["order"] == "AB"]
    ba = [value for value, row in zip(speedups, pairs) if row["order"] == "BA"]
    return {
        "type": "paired_summary",
        "phase": "training_optimizer_update",
        "T": first["T"],
        "B": first["B"],
        "grad_accum": first["grad_accum"],
        "tokens_per_update": first["tokens_per_update"],
        "actual_rate": first["actual_rate"],
        "pairs": len(pairs),
        "ab_pairs": len(ab),
        "ba_pairs": len(ba),
        "order_balanced": abs(len(ab) - len(ba)) <= 1,
        "paired_estimator": "median_of_within_pair_speedup_ratios",
        "paired_speedup_median": float(statistics.median(speedups)),
        "paired_speedup_min": min(speedups),
        "paired_speedup_max": max(speedups),
        "paired_speedup_p25": _quantile(speedups, 0.25),
        "paired_speedup_p75": _quantile(speedups, 0.75),
        "ab_speedup_median": float(statistics.median(ab)) if ab else None,
        "ba_speedup_median": float(statistics.median(ba)) if ba else None,
        "baseline_median_ms": float(statistics.median(baseline_ms)),
        "fused_median_ms": float(statistics.median(frozen_ms)),
        "baseline_resident_allocated_bytes": int(statistics.median(
            row["baseline_resident_allocated_bytes"] for row in pairs)),
        "fused_resident_allocated_bytes": int(statistics.median(
            row["fused_resident_allocated_bytes"] for row in pairs)),
        "baseline_resident_reserved_bytes": int(statistics.median(
            row["baseline_resident_reserved_bytes"] for row in pairs)),
        "fused_resident_reserved_bytes": int(statistics.median(
            row["fused_resident_reserved_bytes"] for row in pairs)),
        "baseline_peak_allocated_bytes": int(statistics.median(
            row["baseline_peak_allocated_bytes"] for row in pairs)),
        "fused_peak_allocated_bytes": int(statistics.median(
            row["fused_peak_allocated_bytes"] for row in pairs)),
        "baseline_peak_reserved_bytes": int(statistics.median(
            row["baseline_peak_reserved_bytes"] for row in pairs)),
        "fused_peak_reserved_bytes": int(statistics.median(
            row["fused_peak_reserved_bytes"] for row in pairs)),
        "fused_pattern_bytes": int(statistics.median(
            row["fused_pattern_bytes"] for row in pairs)),
        "flash_sdpa_forced": all(row["flash_sdpa_forced"] for row in pairs),
        "fused_kernel_verified": all(row["fused_kernel_verified"] for row in pairs),
        "optimizer_step_included": True,
        "data_loading_included": False,
        "selector_fit_included": False,
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--frozen-ckpt", required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--tokens-per-update", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("ABBA requires at least two repeats")
    if args.B * args.T * args.grad_accum != args.tokens_per_update:
        parser.error("B*T*grad_accum must equal tokens-per-update")

    ctx = mp.get_context("spawn")
    pairs = []
    shared = vars(args)
    for repeat in range(args.repeats):
        baseline = {
            **shared, "ckpt": args.baseline_ckpt,
            "variant": "baseline", "repeat": repeat,
        }
        frozen = {
            **shared, "ckpt": args.frozen_ckpt,
            "variant": "frozen", "repeat": repeat,
        }
        order = "AB" if repeat % 2 == 0 else "BA"
        sequence = (("baseline", baseline), ("frozen", frozen))
        if order == "BA":
            sequence = tuple(reversed(sequence))
        results = {}
        for label, spec in sequence:
            gc.collect()
            row = _spawn(ctx, spec)
            results[label] = row
            print(json.dumps(row, sort_keys=True), flush=True)
        pair = _paired_run(results["baseline"], results["frozen"], order)
        pairs.append(pair)
        print(json.dumps(pair, sort_keys=True), flush=True)
    print(json.dumps(_paired_summary(pairs), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

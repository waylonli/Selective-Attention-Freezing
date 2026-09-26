"""ABBA prefill timing for an actual full/frozen checkpoint pair.

Unlike ``bench_inference_triton.py`` (a shape/rate microbenchmark), this script
loads the learned weights, selected head mask, and fitted alpha/rho priors from
formal experiment checkpoints. It measures square causal prefill, not KV-cache
decode. Every ABBA leg uses a fresh spawned process.
"""

from __future__ import annotations

import argparse
import contextlib
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
from bench_inference_triton import (_attn_modules, _gib, _paired_run,  # noqa: E402
                                   _paired_summary, _pattern_bytes, _quantile)
from eval_nanogpt_logprobs import load_model, sha256_file  # noqa: E402


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(spec["seed"])
    torch.cuda.manual_seed_all(spec["seed"])
    model, cfg, checkpoint = load_model(spec["ckpt"], "cuda")
    # load_model returns the CPU checkpoint metadata as well as the live model.
    # It is not needed during timing and can otherwise distort host residency.
    del checkpoint
    if spec["T"] > cfg.block_size:
        raise ValueError(f"T={spec['T']} exceeds checkpoint block_size={cfg.block_size}")
    model.to(dtype=torch.bfloat16).eval()
    modules = _attn_modules(model)
    frozen = sum(int(module.dyn_frozen.sum()) for module in modules)
    total_heads = len(modules) * cfg.n_head
    actual_rate = frozen / total_heads
    if spec["variant"] == "baseline" and frozen:
        raise RuntimeError(f"baseline checkpoint contains {frozen} frozen heads")
    if spec["variant"] == "frozen" and not frozen:
        raise RuntimeError("frozen checkpoint contains no frozen heads")
    generator = torch.Generator(device="cuda").manual_seed(spec["seed"] + 1)
    x = torch.randint(0, cfg.vocab_size, (spec["B"], spec["T"]),
                      device="cuda", generator=generator)
    fused_layers = sum(bool(module.dyn_frozen.any()) for module in modules)
    fused_calls = 0

    def forward() -> None:
        model(x)

    with (torch.inference_mode(),
          torch.autocast(device_type="cuda", dtype=torch.bfloat16),
          sdpa_kernel([SDPBackend.FLASH_ATTENTION])):
        if spec["variant"] == "frozen":
            original = model_mod.fused_mixed_attn

            def counted(*args: Any, **kwargs: Any) -> Any:
                nonlocal fused_calls
                fused_calls += 1
                return original(*args, **kwargs)

            model_mod.fused_mixed_attn = counted
            try:
                forward()
            finally:
                model_mod.fused_mixed_attn = original
            if fused_calls != fused_layers:
                raise RuntimeError(
                    f"expected {fused_layers} fused calls, observed {fused_calls}")
        else:
            forward()
        for _ in range(spec["warmup"]):
            forward()
        torch.cuda.synchronize()

        graph_capture_ms = None
        step = forward
        if spec["cuda_graph"]:
            # CUDA graphs capture on a side stream. Warm every library used by
            # the model on that stream before capture so cuBLAS/SDPA do not try
            # to initialize handles during capture.
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.cuda.graph(graph):
                graph_output = model(x)
            end.record()
            torch.cuda.synchronize()
            graph_capture_ms = float(begin.elapsed_time(end))
            step = graph.replay

        resident_allocated = int(torch.cuda.memory_allocated())
        resident_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(spec["iters"])]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(spec["iters"])]
        for begin, end in zip(starts, ends):
            begin.record(); step(); end.record()
        torch.cuda.synchronize()

    samples = [float(begin.elapsed_time(end)) for begin, end in zip(starts, ends)]
    median_ms = float(statistics.median(samples))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    props = torch.cuda.get_device_properties(0)
    return {
        "type": "run",
        "phase": "prefill",
        "variant": "flash_sdpa" if spec["variant"] == "baseline" else "fused_frozen",
        "checkpoint": spec["ckpt"],
        "checkpoint_sha256": sha256_file(spec["ckpt"]),
        "T": spec["T"], "B": spec["B"],
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
        "requested_rate": actual_rate, "actual_rate": actual_rate,
        "repeat": spec["repeat"], "seed": spec["seed"],
        "median_ms": median_ms, "mean_ms": float(statistics.fmean(samples)),
        "p10_ms": _quantile(samples, 0.1), "p90_ms": _quantile(samples, 0.9),
        "tokens_per_second": spec["B"] * spec["T"] / (median_ms / 1000),
        "samples_ms": samples,
        "resident_allocated_bytes": resident_allocated,
        "resident_reserved_bytes": resident_reserved,
        "incremental_peak_bytes": max(0, peak_allocated - resident_allocated),
        "pattern_bytes": _pattern_bytes(model),
        "freeze_pattern_dtype": cfg.freeze_pattern_dtype,
        "prior_repr": cfg.rand_attn_prior_repr,
        "flash_sdpa_forced": True,
        "fused_kernel_verified": spec["variant"] == "frozen" and fused_calls == fused_layers,
        "expected_fused_layers": fused_layers,
        "fused_calls_in_validation": fused_calls,
        "execution_mode": "cuda_graph" if spec["cuda_graph"] else "eager",
        "graph_capture_ms": graph_capture_ms,
        "device": props.name,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "isolation": "fresh_spawned_worker_per_pair_leg",
    }


def _entry(spec: dict[str, Any], send: Any) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = _worker(spec)
        send.send({"ok": True, "result": result})
    except BaseException:
        send.send({"ok": False, "traceback": traceback.format_exc(),
                   "log": (stdout.getvalue() + stderr.getvalue())[-8000:]})
    finally:
        send.close()


def _spawn(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    recv, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_entry, args=(spec, send))
    process.start(); send.close()
    message = recv.recv(); recv.close(); process.join()
    if not message["ok"]:
        raise RuntimeError(message["traceback"] + "\n" + message["log"])
    if process.exitcode:
        raise RuntimeError(f"worker exit code {process.exitcode}")
    return message["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--frozen-ckpt", required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("ABBA requires at least two repeats")
    ctx = mp.get_context("spawn")
    pairs = []
    for repeat in range(args.repeats):
        base = {**vars(args), "ckpt": args.baseline_ckpt,
                "variant": "baseline", "repeat": repeat}
        frozen = {**vars(args), "ckpt": args.frozen_ckpt,
                  "variant": "frozen", "repeat": repeat}
        order = "AB" if repeat % 2 == 0 else "BA"
        results = {}
        for label, spec in (("baseline", base), ("fused", frozen)) \
                if order == "AB" else (("fused", frozen), ("baseline", base)):
            row = _spawn(ctx, spec)
            results[label] = row
            print(json.dumps(row, sort_keys=True), flush=True)
        pair = _paired_run(results["baseline"], results["fused"], order)
        pair["execution_mode"] = "cuda_graph" if args.cuda_graph else "eager"
        pairs.append(pair)
        print(json.dumps(pair, sort_keys=True), flush=True)
    summary = _paired_summary(pairs)
    summary["execution_mode"] = "cuda_graph" if args.cuda_graph else "eager"
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

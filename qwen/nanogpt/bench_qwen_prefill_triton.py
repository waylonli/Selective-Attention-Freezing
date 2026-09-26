"""Benchmark frozen attention in Qwen3.5 causal prefill.

This benchmark deliberately measures *prefill*, not autoregressive decode:

* inputs are dense, unpadded ``(batch, context)`` token tensors;
* ``use_cache=False`` and no attention mask is supplied;
* the timed output is the backbone ``last_hidden_state``, avoiding the large
  vocabulary projection that would otherwise hide attention-kernel changes.

The baseline uses PyTorch SDPA with the Flash backend forced when the installed
PyTorch exposes that control.  Hybrid variants replace only the standard
softmax-attention layers discovered by ``test_prior_pretrained``; Qwen3.5's
GatedDeltaNet layers are left unchanged.  Uniform causal patterns are used so
that this is a systems benchmark, not a quality experiment.

Example::

    uv run --active python nanogpt/bench_qwen_prefill_triton.py \
      --model Qwen/Qwen3-4B \
      --contexts 512,1024,2048,4096 --rates 0.25,0.5,0.75 \
      --backends torch-split,triton-fused --output results/qwen_prefill.jsonl

Each result is printed as one JSON object and, when ``--output`` is set,
appended to that JSONL file.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hybrid_prior_attention import (  # noqa: E402
    convert_targets_to_hybrid,
    restore_originals,
)
from test_prior_pretrained import (  # noqa: E402
    _iter_blocks,
    discover_softmax_attention,
)


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("rates must be comma-separated values in (0, 1)")
    return values


def _csv_backends(value: str) -> list[str]:
    allowed = {"torch-split", "triton-fused"}
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(values) - allowed
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"backends must be drawn from {sorted(allowed)}; got {sorted(unknown)}")
    return values


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _flash_sdpa_context():
    """Force Flash SDPA when the modern PyTorch API is available.

    The context also covers the torch-split hybrid's live heads.  Triton-fused
    heads do not call SDPA, while the rest of the model is unchanged.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    except (ImportError, AttributeError):
        # Older PyTorch fallback.  This API is deprecated in newer releases,
        # hence the preference for torch.nn.attention.sdpa_kernel above.
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
                enable_cudnn=False,
            )
        return contextlib.nullcontext()


def _uniform_causal_priors(
    n_layers: int,
    n_heads: int,
    context: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a broadcast view of one row-stochastic causal pattern.

    The conversion code materialises only the selected heads.  Keeping the
    leading layer/head dimensions as a view avoids allocating a second full
    ``n_layers * n_heads * T^2`` tensor before conversion.
    """
    pattern = torch.ones((context, context), device=device, dtype=dtype).tril_()
    denominators = torch.arange(
        1, context + 1, device=device, dtype=torch.float32).to(dtype)
    pattern.div_(denominators[:, None])
    return pattern.view(1, 1, context, context).expand(
        n_layers, n_heads, context, context)


def _nested_head_masks(
    rate: float,
    n_layers: int,
    n_heads: int,
    seed: int,
) -> list[torch.Tensor]:
    """Deterministic per-layer masks, nested as the requested rate grows."""
    n_frozen = int(round(rate * n_heads))
    n_frozen = min(max(n_frozen, 1), n_heads - 1)
    masks = []
    for layer_idx in range(n_layers):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 104729 * layer_idx)
        order = torch.randperm(n_heads, generator=generator)
        mask = torch.zeros(n_heads, dtype=torch.bool)
        mask[order[:n_frozen]] = True
        masks.append(mask)
    return masks


def _hybrid_modules(model, targets) -> list[torch.nn.Module]:
    blocks = _iter_blocks(model)
    return [getattr(blocks[target.layer_idx], target.attr_name) for target in targets]


def _call_counts(modules: Iterable[torch.nn.Module]) -> tuple[int, int]:
    hybrid = sum(int(getattr(module, "hybrid_calls", 0)) for module in modules)
    triton = sum(int(getattr(module, "triton_calls", 0)) for module in modules)
    return hybrid, triton


def _backbone_forward(backbone, input_ids: torch.Tensor) -> torch.Tensor:
    output = backbone(
        input_ids=input_ids,
        attention_mask=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return output.last_hidden_state


class AttnOpProbe:
    """Times ONLY the attention operator calls of a forward: every
    `torch.nn.functional.scaled_dot_product_attention` (HF flash/SDPA path,
    live heads of the split fallback) and every `fused_mixed_attn` launch
    (hybrid Triton path). CUDA-event pairs around each call, summed per
    forward — the in-situ "attention op" scope of kernel/bench_attn_only.py."""

    def __init__(self):
        import torch.nn.functional as F
        import hybrid_prior_attention as HPA
        self._F, self._HPA = F, HPA
        self._orig = (F.scaled_dot_product_attention, HPA.fused_mixed_attn)
        self.events: list[tuple] = []

    def _wrap(self, fn):
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return out
        return wrapped

    def __enter__(self):
        self._F.scaled_dot_product_attention = self._wrap(self._orig[0])
        self._HPA.fused_mixed_attn = self._wrap(self._orig[1])
        return self

    def __exit__(self, *exc):
        self._F.scaled_dot_product_attention = self._orig[0]
        self._HPA.fused_mixed_attn = self._orig[1]

    def reset(self):
        self.events.clear()

    def total_ms(self) -> float:
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.events)


@torch.inference_mode()
def _benchmark(
    backbone,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[dict, torch.Tensor]:
    """Time GPU execution with CUDA events and return a small output probe."""
    samples_ms: list[float] = []
    op_samples_ms: list[float] = []
    last_hidden = None
    with _flash_sdpa_context(), AttnOpProbe() as probe:
        # One untimed call before per-repeat warmups catches model-level lazy
        # setup; Triton JIT compilation is likewise kept outside measurements.
        last_hidden = _backbone_forward(backbone, input_ids)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(repeats):
            for _ in range(warmup):
                last_hidden = _backbone_forward(backbone, input_ids)
            torch.cuda.synchronize()

            for _ in range(iters):
                probe.reset()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                last_hidden = _backbone_forward(backbone, input_ids)
                end.record()
                end.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))
                op_samples_ms.append(probe.total_ms())

    assert last_hidden is not None
    # Probe the beginning, middle, and end.  It is copied only after timing and
    # is sufficient to compare the torch and Triton implementations without
    # retaining a B*T*hidden float32 tensor for every variant.
    positions = sorted({0, input_ids.size(1) // 2, input_ids.size(1) - 1})
    probe = last_hidden[:, positions].float().cpu()
    median_ms = statistics.median(samples_ms)
    result = {
        "median_ms": median_ms,
        "attn_op_median_ms": statistics.median(op_samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p10_ms": _percentile(samples_ms, 0.10),
        "p90_ms": _percentile(samples_ms, 0.90),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples": len(samples_ms),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }
    return result, probe


class JsonlEmitter:
    def __init__(self, output: str | None):
        self.path = Path(output) if output else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A benchmark invocation is one coherent result set.  Truncate an
            # old file instead of silently mixing hardware/software runs.
            self.path.write_text("", encoding="utf-8")

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 frozen-attention causal-prefill benchmark")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Local Hugging Face model path (or model id with --no-offline)",
    )
    parser.add_argument("--contexts", type=_csv_ints, default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rates", type=_csv_floats, default=[0.25, 0.5, 0.75])
    parser.add_argument(
        "--repr", choices=["dense", "decomposed"], default="dense",
        help="frozen-prior representation: dense (T,T) patterns or the "
             "decomposed alpha+rho vectors (alpha = rho = 0 gives the same "
             "uniform causal pattern, so the two are directly comparable)")
    parser.add_argument(
        "--backends", type=_csv_backends,
        default=["torch-split", "triton-fused"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument(
        "--offline", action=argparse.BooleanOptionalAction, default=True,
        help="Use only local Hugging Face files (default: true)",
    )
    parser.add_argument(
        "--check-correctness", action=argparse.BooleanOptionalAction, default=True,
        help="Compare fixed-position hidden states for torch-split vs Triton",
    )
    parser.add_argument(
        "--strict-triton", action=argparse.BooleanOptionalAction, default=True,
        help="Fail if a requested Triton variant takes the fallback path",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("batch-size/iters/repeats must be positive and warmup non-negative")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    emit = JsonlEmitter(args.output)

    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.offline,
        "torch_dtype": dtype,
        "attn_implementation": "sdpa",
    }
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
    model.eval()
    model.config.use_cache = False
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError(f"{type(model).__name__} has no .model backbone")

    targets = discover_softmax_attention(model)
    if not targets:
        raise RuntimeError("No standard softmax-attention layers were discovered")
    n_heads = int(model.config.num_attention_heads)
    n_blocks = int(model.config.num_hidden_layers)
    head_dim = int(getattr(
        model.config, "head_dim", model.config.hidden_size // n_heads))
    attention_classes = sorted({type(target.module).__name__ for target in targets})
    if attention_classes not in (["Qwen3_5Attention"], ["Qwen3Attention"]):
        raise RuntimeError(
            "The hybrid adapter supports Qwen3 / Qwen3.5 only; discovered "
            f"{attention_classes}")

    flash_available = bool(getattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: False)())
    metadata = {
        "type": "metadata",
        "benchmark": "causal_prefill_backbone",
        "decode_or_prefill": "prefill",
        "use_cache": False,
        "padding": False,
        "pattern_content": "uniform_causal_systems_only",
        "prior_repr": args.repr,
        "model": args.model,
        "model_class": type(model).__name__,
        "params_b": sum(parameter.numel() for parameter in model.parameters()) / 1e9,
        "load_seconds": time.time() - load_start,
        "blocks": n_blocks,
        "softmax_attention_layers": len(targets),
        "softmax_attention_layer_indices": [target.layer_idx for target in targets],
        "attention_classes": attention_classes,
        "query_heads": n_heads,
        "kv_heads": int(getattr(model.config, "num_key_value_heads", n_heads)),
        "head_dim": head_dim,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "flash_sdpa_available": flash_available,
        "contexts": args.contexts,
        "batch_size": args.batch_size,
        "rates": args.rates,
        "backends": args.backends,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    emit(metadata)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    max_context = max(args.contexts)
    all_input_ids = torch.randint(
        low=0,
        high=int(model.config.vocab_size),
        size=(args.batch_size, max_context),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    for context in args.contexts:
        input_ids = all_input_ids[:, :context].contiguous()
        baseline_perf, baseline_probe = _benchmark(
            backbone, input_ids,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        baseline_record = {
            "type": "result",
            "backend": "flash-sdpa",
            "rate": 0.0,
            "context": context,
            "batch_size": args.batch_size,
            "tokens": args.batch_size * context,
            "speedup_vs_flash": 1.0,
            **baseline_perf,
        }
        baseline_record["tokens_per_second"] = (
            baseline_record["tokens"] / (baseline_perf["median_ms"] / 1000.0))
        emit(baseline_record)

        if args.repr == "decomposed":
            # alpha = rho = 0  ->  softmax(0) = uniform causal: identical
            # content to the dense uniform prior, O(T) storage
            priors = None
            zeros = torch.zeros(n_heads, context, device=device)
            decomp = [(zeros, zeros)] * len(targets)
        else:
            priors = _uniform_causal_priors(
                len(targets), n_heads, context, device=device, dtype=dtype)
            decomp = None

        for rate in args.rates:
            masks = _nested_head_masks(rate, len(targets), n_heads, args.seed)
            frozen_per_layer = [int(mask.sum()) for mask in masks]
            effective_rate = sum(frozen_per_layer) / (len(targets) * n_heads)
            mask_indices = [
                torch.nonzero(mask, as_tuple=True)[0].tolist() for mask in masks]
            backend_probes: dict[str, torch.Tensor] = {}

            for backend_name in args.backends:
                use_triton = backend_name == "triton-fused"
                saved = convert_targets_to_hybrid(
                    model, targets, priors, masks, device, dtype,
                    use_triton=use_triton,
                    assume_causal_prefill=True,
                    decomp=decomp,
                )
                modules = _hybrid_modules(model, targets)
                before_hybrid, before_triton = _call_counts(modules)
                try:
                    perf, probe = _benchmark(
                        backbone, input_ids,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    after_hybrid, after_triton = _call_counts(modules)
                    hybrid_delta = after_hybrid - before_hybrid
                    triton_delta = after_triton - before_triton
                    if use_triton and args.strict_triton:
                        if triton_delta <= 0 or triton_delta != hybrid_delta:
                            raise RuntimeError(
                                "Triton was requested but not used for every converted "
                                f"attention call: triton={triton_delta}, "
                                f"hybrid={hybrid_delta}. This usually means Qwen passed "
                                "a mask/cache or the kernel rejected the shape.")

                    record = {
                        "type": "result",
                        "backend": backend_name,
                        "rate": rate,
                        "effective_rate": effective_rate,
                        "frozen_heads_per_softmax_layer": frozen_per_layer,
                        "frozen_head_indices": mask_indices,
                        "context": context,
                        "batch_size": args.batch_size,
                        "tokens": args.batch_size * context,
                        "speedup_vs_flash": baseline_perf["median_ms"] / perf["median_ms"],
                        "attn_op_speedup_vs_flash": (
                            baseline_perf["attn_op_median_ms"] / perf["attn_op_median_ms"]),
                        "hybrid_calls": hybrid_delta,
                        "triton_calls": triton_delta,
                        **perf,
                    }
                    record["tokens_per_second"] = (
                        record["tokens"] / (perf["median_ms"] / 1000.0))
                    probe_delta = probe - baseline_probe
                    record["probe_rms_diff_vs_flash"] = float(
                        probe_delta.square().mean().sqrt())
                    emit(record)
                    backend_probes[backend_name] = probe
                finally:
                    restore_originals(saved)
                    del modules, saved
                    gc.collect()
                    torch.cuda.empty_cache()

            if args.check_correctness and {
                    "torch-split", "triton-fused"}.issubset(backend_probes):
                torch_probe = backend_probes["torch-split"]
                triton_probe = backend_probes["triton-fused"]
                delta = triton_probe - torch_probe
                reference_rms = float(torch_probe.square().mean().sqrt())
                emit({
                    "type": "correctness",
                    "context": context,
                    "rate": rate,
                    "effective_rate": effective_rate,
                    "probe_positions": [0, context // 2, context - 1],
                    "max_abs_diff_triton_vs_torch": float(delta.abs().max()),
                    "mean_abs_diff_triton_vs_torch": float(delta.abs().mean()),
                    "rms_diff_triton_vs_torch": float(delta.square().mean().sqrt()),
                    "relative_rms_diff_triton_vs_torch": (
                        float(delta.square().mean().sqrt()) /
                        max(reference_rms, 1e-12)
                    ),
                })

        del priors
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

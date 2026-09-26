"""Matched finetuning-throughput benchmark on real downstream-task batches.

This intentionally does not report task quality. Each checkpoint sees the same
ordered batches and performs the same number of optimizer updates, making step
time comparable across full and frozen models. A warmup phase absorbs optimizer
state allocation and Triton/SDPA compilation. Gradient accumulation is measured
as part of one complete optimizer update.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

os.environ["FROZEN_ATTN_FUSED"] = "1"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_nanogpt_logprobs import load_model
import model as model_impl
from kernel import fused_attn as fused_attn_impl
from finetune_downstream import (
    PromptDataset,
    TASKS,
    answer_logits,
    autocast,
    collate,
    stratified_train_dev_indices,
)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def frozen_count(model):
    return sum(
        int(block.main_block.dyn_frozen.sum())
        for block in model.transformer.h
        if hasattr(block.main_block, "dyn_frozen")
    )


def attention_modules(model):
    return [block.main_block for block in model.transformer.h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--timed_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--expected_source_iter", type=int, default=None)
    parser.add_argument("--expected_frozen_heads", type=int, default=None)
    parser.add_argument("--expected_block_size", type=int, default=None)
    parser.add_argument("--expected_position_encoding", default=None)
    parser.add_argument("--require_fused_frozen", action="store_true")
    args = parser.parse_args()
    if (args.warmup_steps < 1 or args.timed_steps < 1
            or args.batch_size < 1 or args.grad_accum < 1):
        raise ValueError(
            "warmup_steps, timed_steps, batch_size, and grad_accum must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    model, config, source_checkpoint = load_model(args.ckpt, args.device)
    source_iter = source_checkpoint.get("iter_num")
    del source_checkpoint
    gc.collect()
    model.train()
    initial_frozen_heads = frozen_count(model)
    if args.expected_source_iter is not None:
        assert source_iter == args.expected_source_iter, (
            source_iter, args.expected_source_iter)
    if args.expected_frozen_heads is not None:
        assert initial_frozen_heads == args.expected_frozen_heads, (
            initial_frozen_heads, args.expected_frozen_heads)
    if args.expected_block_size is not None:
        assert config.block_size == args.expected_block_size, (
            config.block_size, args.expected_block_size)
    if args.expected_position_encoding is not None:
        assert config.position_encoding == args.expected_position_encoding, (
            config.position_encoding, args.expected_position_encoding)

    modules = attention_modules(model)
    expected_fused_layers = sum(
        int(bool(module.dyn_frozen.any().item())) for module in modules
    )
    fused_ready = bool(
        model_impl._FUSED
        and fused_attn_impl.HAVE_TRITON
        and all(module.attn_mask_rate == 0 for module in modules)
        and all(module.dropout == 0.0 for module in modules)
    )
    if args.require_fused_frozen and initial_frozen_heads:
        assert fused_ready, "frozen checkpoints would not use the fused training path"

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset_args = TASKS[args.task]["dataset"]
    raw = load_dataset(*dataset_args)
    train_indices, _ = stratified_train_dev_indices(
        raw["train"], args.task, dev_fraction=0.1, seed=args.seed)
    max_length = config.block_size if args.max_length is None else args.max_length
    if max_length < 1 or max_length > config.block_size:
        raise ValueError(
            f"max_length must be in [1, {config.block_size}], got {max_length}")
    train_set = PromptDataset(
        raw["train"], args.task, tokenizer, max_length,
        indices=train_indices, split_name="train")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        drop_last=True,
        collate_fn=lambda rows: collate(rows, tokenizer.eot_token))
    iterator = iter(loader)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    compute_times = []
    pipeline_times = []
    padded_tokens = []
    real_tokens = []
    batch_lengths = []
    batch_sizes = []
    batch_digest = hashlib.sha256()
    total_steps = args.warmup_steps + args.timed_steps
    fused_dispatch_calls = 0
    fused_dispatch_verified = expected_fused_layers == 0
    for step in range(total_steps):
        pipeline_start = time.perf_counter()
        update_batches = []
        update_padded_tokens = 0
        update_real_tokens = 0
        update_examples = 0
        update_lengths = []
        update_signatures = []
        for micro_step in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            inputs = batch["input_ids"]
            lengths = batch["lengths"]
            update_batches.append(batch)
            update_padded_tokens += int(inputs.numel())
            update_real_tokens += int(lengths.sum())
            update_examples += int(inputs.size(0))
            update_lengths.append(int(inputs.shape[1]))
            update_signatures.append({
                "micro_step": micro_step,
                "length": int(inputs.shape[1]),
                "source_indices": [
                    int(item["source_index"]) for item in batch["metadata"]
                ],
            })
        torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        original_fused = None
        if step == 0 and expected_fused_layers:
            original_fused = model_impl.fused_mixed_attn

            def counted_fused(*fused_args, **fused_kwargs):
                nonlocal fused_dispatch_calls
                fused_dispatch_calls += 1
                return original_fused(*fused_args, **fused_kwargs)

            model_impl.fused_mixed_attn = counted_fused
        try:
            for batch in update_batches:
                inputs = batch["input_ids"].to(device)
                lengths = batch["lengths"].to(device)
                targets = batch["targets"].to(device)
                # Fail loudly if live heads cannot use Flash. Frozen heads
                # bypass SDPA and are required to dispatch Triton.
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    with autocast(device, dtype):
                        logits = answer_logits(model, inputs, lengths)
                        loss = F.cross_entropy(logits, targets) / args.grad_accum
                loss.backward()
        finally:
            if original_fused is not None:
                model_impl.fused_mixed_attn = original_fused
        if step == 0 and expected_fused_layers:
            expected_calls = expected_fused_layers * args.grad_accum
            fused_dispatch_verified = fused_dispatch_calls == expected_calls
            assert fused_dispatch_verified, (
                f"expected {expected_calls} fused calls, "
                f"observed {fused_dispatch_calls} calls"
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        compute_end = time.perf_counter()

        if step + 1 == args.warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)
        if step >= args.warmup_steps:
            compute_times.append(compute_end - compute_start)
            pipeline_times.append(compute_end - pipeline_start)
            padded_tokens.append(update_padded_tokens)
            real_tokens.append(update_real_tokens)
            batch_lengths.extend(update_lengths)
            batch_sizes.append(update_examples)
            batch_digest.update(
                json.dumps(update_signatures, sort_keys=True).encode("utf-8"))

    n_examples = sum(batch_sizes)
    compute_total = sum(compute_times)
    pipeline_total = sum(pipeline_times)
    result = {
        "checkpoint": str(pathlib.Path(args.ckpt).resolve()),
        "source_iter": source_iter,
        "task": args.task,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "micro_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": max_length,
        "drop_last": True,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "dtype": args.dtype,
        "frozen_heads": frozen_count(model),
        "fused_backend_ready": fused_ready,
        "fused_backend_required": bool(
            args.require_fused_frozen and initial_frozen_heads),
        "flash_sdpa_forced": True,
        "expected_fused_layers": expected_fused_layers,
        "fused_dispatch_calls_in_validation": fused_dispatch_calls,
        "fused_dispatch_verified": fused_dispatch_verified,
        "compute_step_ms_mean": 1000 * compute_total / args.timed_steps,
        "compute_step_ms_median": 1000 * quantile(compute_times, 0.5),
        "pipeline_step_ms_mean": 1000 * pipeline_total / args.timed_steps,
        "pipeline_step_ms_median": 1000 * quantile(pipeline_times, 0.5),
        "compute_examples_per_second": n_examples / compute_total,
        "pipeline_examples_per_second": n_examples / pipeline_total,
        "compute_padded_tokens_per_second": sum(padded_tokens) / compute_total,
        "pipeline_padded_tokens_per_second": sum(padded_tokens) / pipeline_total,
        "real_tokens_per_second": sum(real_tokens) / pipeline_total,
        "batch_length_p10_p50_p90": [
            quantile(batch_lengths, q) for q in (0.1, 0.5, 0.9)],
        "batch_order_sha256": batch_digest.hexdigest(),
        "compute_step_ms": [1000 * value for value in compute_times],
        "pipeline_step_ms": [1000 * value for value in pipeline_times],
        "batch_lengths": batch_lengths,
        "total_padded_tokens": sum(padded_tokens),
        "total_real_tokens": sum(real_tokens),
        "peak_vram_mb_after_warmup": torch.cuda.max_memory_allocated(device) / 1e6,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

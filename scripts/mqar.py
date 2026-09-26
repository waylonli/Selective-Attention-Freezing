"""Adapt or evaluate a checkpoint on the paper's MQAR protocol."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from nanogpt.eval_nanogpt_logprobs import load_model
from nanogpt.finetune_downstream import capture_frozen_state, frozen_state_matches
from scripts.export_checkpoint import sha256
from zoology.data.multiquery_ar import multiquery_ar


def data(seed, length, pairs, count):
    previous = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            segment = multiquery_ar(8192, count, length, seed, num_kv_pairs=pairs,
                                    power_a=.01, random_non_queries=True)
    finally:
        np.random.set_state(previous)
    x, y = segment.inputs, segment.labels
    if x.shape != (count, length) or not ((y != -100).sum(1) == pairs).all():
        raise ValueError("Unexpected MQAR generator output")
    return x, y


def logits(model, x, labels):
    mask = labels != -100
    return model.lm_head(model.forward_hidden(x)[mask]), labels[mask]


def context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()


def backend(device):
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.startswith("cuda") else nullcontext()


@torch.no_grad()
def evaluate(model, x, y, batch, device):
    model.eval()
    correct, queries, nll = 0, 0, 0.
    for start in range(0, len(x), batch):
        with backend(device), context(device):
            scores, labels = logits(model, x[start:start+batch].to(device), y[start:start+batch].to(device))
        correct += int((scores.argmax(-1) == labels).sum())
        queries += labels.numel()
        nll += float(F.cross_entropy(scores.float(), labels, reduction="sum"))
    return {"accuracy": correct / queries, "correct": correct, "queries": queries, "nll": nll / queries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--train-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.seed < 1337:
        parser.error("Task seeds start at 1337 in this protocol")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model, config, checkpoint = load_model(args.ckpt, args.device)
    if args.length > config.block_size or config.vocab_size < 8192:
        parser.error("The checkpoint cannot represent this task configuration")
    args.out.mkdir(parents=True)
    frozen = capture_frozen_state(model)
    if args.action == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        stream = hashlib.sha256()
        for step in range(args.updates):
            x, y = data(1000000 + 100000 * (args.seed - 1337) + step, 512, 8, args.train_batch)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())
            model.train()
            with backend(args.device), context(args.device):
                scores, labels = logits(model, x.to(args.device), y.to(args.device))
                loss = F.cross_entropy(scores.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 250 == 0:
                print(f"update={step + 1} loss={loss.item():.5f}", flush=True)
        torch.save({"model": model.state_dict(), "model_args": asdict(config),
                    "attention_layout": checkpoint.get("attention_layout"), "task": "mqar",
                    "updates": args.updates, "source_checkpoint_sha256": sha256(args.ckpt)},
                   args.out / "finetuned.pt")
    results = []
    for pairs in args.pairs:
        x, y = data(800001 + args.length + pairs, args.length, pairs, args.examples)
        results.append({"length": args.length, "pairs": pairs,
                        **evaluate(model, x, y, args.eval_batch, args.device),
                        "inputs_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                        "labels_sha256": hashlib.sha256(y.numpy().tobytes()).hexdigest()})
    if not frozen_state_matches(model, frozen):
        raise RuntimeError("Fixed patterns changed")
    record = {"source_sha256": sha256(args.ckpt), "task_seed": args.seed, "results": results}
    if args.action == "train":
        record["training_data_sha256"] = stream.hexdigest()
    (args.out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

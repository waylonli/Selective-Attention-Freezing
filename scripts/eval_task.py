"""Evaluate a saved finetuned checkpoint using the original answer-token scoring."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from nanogpt.eval_nanogpt_logprobs import load_model
from nanogpt.finetune_downstream import TASKS, PromptDataset, collate, evaluate
from scripts.export_checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    model, config, checkpoint = load_model(args.ckpt, args.device)
    metadata = checkpoint.get("task_finetune", {})
    task = args.task or metadata.get("task")
    length = args.max_length or metadata.get("max_length")
    if task not in TASKS or length is None:
        parser.error("Specify --task and --max-length when absent from the checkpoint")
    if length > config.block_size:
        parser.error("Task length exceeds the checkpoint context")
    tokenizer = tiktoken.get_encoding("gpt2")
    raw = load_dataset(*TASKS[task]["dataset"], split="validation")
    dataset = PromptDataset(raw, task, tokenizer, length, split_name="validation")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate, pad_id=tokenizer.eot_token))
    choices = [tokenizer.encode(choice) for choice in TASKS[task]["choices"]]
    if not all(len(choice) == 1 for choice in choices):
        raise ValueError("Expected single-token verbaliser choices")
    choice_ids = torch.tensor([choice[0] for choice in choices], device=args.device)
    scores = evaluate(model, loader, choice_ids, args.device, torch.bfloat16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"task": task, "max_length": length,
                                   "checkpoint_sha256": sha256(args.ckpt), "scores": scores}, indent=2) + "\n")


if __name__ == "__main__":
    main()

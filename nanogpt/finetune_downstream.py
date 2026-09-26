"""Task-specific causal-LM finetuning for classification and multiple choice.

No classifier head is added. Only the single answer token is supervised, so
the architecture and frozen-head mechanism remain unchanged.
"""

import argparse
import contextlib
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dynamic_freeze import DynamicFreezeController
from eval_nanogpt_logprobs import load_model


def format_sst2_prompt(row):
    return (f"{row['sentence']}\n"
            "Question: Is this sentence positive or negative?\nAnswer:")


def format_boolq_prompt(row):
    # The canonical dataset normally omits the final question mark. Avoid a
    # duplicated mark if an alternate mirror already includes it.
    question = str(row["question"]).rstrip()
    question = question[:-1] if question.endswith("?") else question
    return f"{row['passage']}\nQuestion: {question}?\nAnswer:"


def format_quality_prompt(row):
    option_lines = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip("ABCD", row["options"])
    )
    return (f"{row['article']}\nQuestion: {row['question'].strip()}\n"
            f"{option_lines}\nAnswer:")


TASKS = {
    "sst2": {
        # Pinned to lm-eval 0.4.12's glue/sst2/default.yaml.
        "dataset": ("nyu-mll/glue", "sst2"),
        "task_version": 1.0,
        "choices": (" negative", " positive"),
        "prompt_template": (
            "{{sentence}}\nQuestion: Is this sentence positive or negative?\n"
            "Answer:"),
        "prompt": format_sst2_prompt,
        "scheduler_max_length": 128,
        "label_field": "label",
        "label_offset": 0,
        "id_field": "idx",
    },
    "boolq": {
        # Pinned to lm-eval 0.4.12's super_glue/boolq/default.yaml.
        "dataset": ("aps/super_glue", "boolq"),
        "task_version": 2.0,
        "choices": (" no", " yes"),
        "prompt_template": "{{passage}}\nQuestion: {{question}}?\nAnswer:",
        "prompt": format_boolq_prompt,
        "scheduler_max_length": 512,
        "label_field": "label",
        "label_offset": 0,
        "id_field": "idx",
    },
    "quality": {
        # QuALITY is a four-way long-document question-answering benchmark.
        "dataset": ("tasksource/QuALITY",),
        "task_version": 1.0,
        "choices": (" A", " B", " C", " D"),
        "prompt_template": (
            "{{article}}\nQuestion: {{question}}\n"
            "A. {{options[0]}}\nB. {{options[1]}}\n"
            "C. {{options[2]}}\nD. {{options[3]}}\nAnswer:"),
        "prompt": format_quality_prompt,
        "scheduler_max_length": 4096,
        "label_field": "gold_label",
        "label_offset": -1,
        "id_field": "question_unique_id",
    },
}


def task_label(row, task):
    spec = TASKS[task]
    label = int(row[spec["label_field"]]) + int(spec["label_offset"])
    if not 0 <= label < len(spec["choices"]):
        raise ValueError(f"invalid label {label} for task {task}")
    return label


def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample_indices(rows, task, limit, seed):
    """Select a deterministic, approximately class-proportional subset."""
    n_rows = len(rows)
    if limit is None or limit >= n_rows:
        return list(range(n_rows))
    if limit <= 0:
        raise ValueError("example limit must be positive")

    groups = defaultdict(list)
    for index in range(n_rows):
        groups[task_label(rows[index], task)].append(index)
    rng = random.Random(seed)
    for label in sorted(groups):
        rng.shuffle(groups[label])

    exact = {
        label: limit * len(indices) / n_rows
        for label, indices in groups.items()
    }
    counts = {label: int(value) for label, value in exact.items()}
    remainder = limit - sum(counts.values())
    order = sorted(groups, key=lambda label: (-(exact[label] - counts[label]), label))
    for label in order[:remainder]:
        counts[label] += 1

    selected = []
    for label in sorted(groups):
        selected.extend(groups[label][:counts[label]])
    return sorted(selected)


def stratified_train_dev_indices(rows, task, dev_fraction, seed, limit=None):
    """Create a deterministic split used only for model/epoch selection."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    selected = stratified_sample_indices(rows, task, limit, seed)
    if len(selected) < 2:
        raise ValueError("at least two training examples are required")

    groups = defaultdict(list)
    for index in selected:
        groups[task_label(rows[index], task)].append(index)
    rng = random.Random(seed + 1)
    train_indices, dev_indices = [], []
    for label in sorted(groups):
        indices = groups[label]
        rng.shuffle(indices)
        n_dev = int(round(len(indices) * dev_fraction))
        if len(indices) > 1:
            n_dev = min(max(n_dev, 1), len(indices) - 1)
        else:
            n_dev = 0
        dev_indices.extend(indices[:n_dev])
        train_indices.extend(indices[n_dev:])

    # Very small smoke subsets can contain singleton classes only. Preserve a
    # non-empty selection split without changing normal full-data behaviour.
    if not dev_indices:
        dev_indices.append(train_indices.pop())
    if not train_indices:
        train_indices.append(dev_indices.pop())
    return sorted(train_indices), sorted(dev_indices)


class PromptDataset(Dataset):
    def __init__(self, rows, task, tokenizer, max_length, limit=None,
                 indices=None, split_name="unknown"):
        spec = TASKS[task]
        answer_ids = [tokenizer.encode(choice) for choice in spec["choices"]]
        if any(len(tokens) != 1 for tokens in answer_ids):
            raise ValueError("task verbalizers must each be one GPT-2 token")
        self.choice_ids = torch.tensor([tokens[0] for tokens in answer_ids])
        self.examples = []
        if indices is not None and limit is not None:
            raise ValueError("pass either indices or limit, not both")
        if indices is None:
            count = min(len(rows), limit) if limit is not None else len(rows)
            indices = range(count)
        for index in indices:
            row = rows[index]
            prompt = spec["prompt"](row)
            all_tokens = tokenizer.encode(prompt)
            tokens = all_tokens[-max_length:]
            if not tokens:
                tokens = [tokenizer.eot_token]
            label = task_label(row, task)
            example_id = row.get(spec["id_field"], index)
            if hasattr(example_id, "item"):
                example_id = example_id.item()
            self.examples.append({
                "tokens": tokens,
                "target": int(self.choice_ids[label]),
                "label": label,
                "example_id": example_id,
                "source_index": int(index),
                "split": split_name,
                "truncated": len(all_tokens) > max_length,
                "original_token_length": len(all_tokens),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(examples, pad_id, fixed_length=None):
    lengths = torch.tensor([len(example["tokens"]) for example in examples])
    padded_length = int(lengths.max()) if fixed_length is None else fixed_length
    if padded_length < int(lengths.max()):
        raise ValueError("fixed_length is shorter than an example")
    inputs = torch.full(
        (len(examples), padded_length), pad_id, dtype=torch.long)
    for row, example in enumerate(examples):
        inputs[row, :len(example["tokens"])] = torch.tensor(example["tokens"])
    return {
        "input_ids": inputs,
        "lengths": lengths,
        "targets": torch.tensor([example["target"] for example in examples]),
        "labels": torch.tensor([example["label"] for example in examples]),
        "metadata": [{key: value for key, value in example.items()
                      if key not in ("tokens", "target", "label")}
                     for example in examples],
    }


class CyclingBatches:
    """Cycle over a dedicated calibration loader without touching train order."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.iterator = iter(loader)

    def __call__(self, split):
        if split != "train":
            raise ValueError("downstream schedulers currently use train calibration only")
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return {
            **batch,
            "input_ids": batch["input_ids"].to(self.device),
            "lengths": batch["lengths"].to(self.device),
            "targets": batch["targets"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }


def autocast(device, dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def answer_logits(model, inputs, lengths):
    hidden = model.forward_hidden(inputs)
    rows = torch.arange(inputs.size(0), device=inputs.device)
    # Only the final real prompt position predicts the answer token. Applying
    # the LM head after gathering avoids materializing (B, T, vocab) logits.
    return model.lm_head(hidden[rows, lengths - 1])


@torch.no_grad()
def evaluate(model, loader, choice_ids, device, dtype, return_examples=False):
    model.eval()
    correct = count = 0
    full_nll = choice_nll = choice_brier = 0.0
    example_results = []
    truncated = 0
    choice_ids = choice_ids.to(device)
    for batch in loader:
        inputs, lengths = batch["input_ids"], batch["lengths"]
        targets, labels = batch["targets"], batch["labels"]
        inputs, lengths = inputs.to(device), lengths.to(device)
        targets, labels = targets.to(device), labels.to(device)
        with autocast(device, dtype):
            logits = answer_logits(model, inputs, lengths).float()
        choices = logits.index_select(-1, choice_ids)
        choice_log_probs = F.log_softmax(choices, dim=-1)
        choice_probs = choice_log_probs.exp()
        predictions = choices.argmax(-1)
        correct += int((predictions == labels).sum())
        count += inputs.size(0)
        choice_nll += float(F.cross_entropy(
            choices, labels, reduction="sum"))
        full_nll += float(F.cross_entropy(
            logits, targets, reduction="sum"))
        one_hot = F.one_hot(labels, num_classes=choices.size(-1)).float()
        choice_brier += float(((choice_probs - one_hot) ** 2).sum(-1).sum())
        truncated += sum(bool(item["truncated"]) for item in batch["metadata"])
        if return_examples:
            full_choice_log_probs = choices - logits.logsumexp(-1, keepdim=True)
            for row, metadata in enumerate(batch["metadata"]):
                example_results.append({
                    **metadata,
                    "label": int(labels[row]),
                    "prediction": int(predictions[row]),
                    "correct": bool(predictions[row] == labels[row]),
                    "choice_logprobs": [
                        float(value) for value in choice_log_probs[row].cpu()],
                    "full_vocab_choice_logprobs": [
                        float(value) for value in full_choice_log_probs[row].cpu()],
                })
    metrics = {
        "accuracy": correct / count,
        "choice_nll": choice_nll / count,
        "choice_brier": choice_brier / count,
        "full_vocab_nll": full_nll / count,
        "n_examples": count,
        "truncated_examples": truncated,
        "truncated_fraction": truncated / count,
    }
    if return_examples:
        metrics["examples"] = example_results
    return metrics


def capture_frozen_state(model):
    """Snapshot the immutable dynamic-freeze mask and selected patterns."""
    snapshots = []
    for layer, block in enumerate(model.transformer.h):
        attention = block.main_block
        if not hasattr(attention, "dyn_frozen"):
            continue
        mask = attention.dyn_frozen.detach().cpu().clone()
        state = {}
        for name in ("dyn_pattern", "dyn_alpha", "dyn_rho", "dyn_z", "dyn_rho_band"):
            value = getattr(attention, name, None)
            if value is not None:
                state[name] = value.detach().cpu().clone()
        snapshots.append({"layer": layer, "mask": mask, "state": state})
    return snapshots


def summarize_frozen_state(snapshots, total_heads):
    heads = []
    for snapshot in snapshots:
        for head in snapshot["mask"].nonzero(as_tuple=True)[0].tolist():
            heads.append(f"L{snapshot['layer']}H{head}")
    return {
        "frozen_heads": heads,
        "frozen_count": len(heads),
        "frozen_rate": len(heads) / total_heads,
    }


def frozen_state_matches(model, snapshots):
    by_layer = {snapshot["layer"]: snapshot for snapshot in snapshots}
    for layer, block in enumerate(model.transformer.h):
        attention = block.main_block
        if not hasattr(attention, "dyn_frozen"):
            continue
        snapshot = by_layer.get(layer)
        if snapshot is None:
            return False
        mask = attention.dyn_frozen.detach().cpu()
        if not torch.equal(mask, snapshot["mask"]):
            return False
        for name, expected in snapshot["state"].items():
            value = getattr(attention, name, None)
            if value is None or not torch.equal(value.detach().cpu(), expected):
                return False
    return len(by_layer) == sum(
        hasattr(block.main_block, "dyn_frozen") for block in model.transformer.h)


def frozen_head_count(model):
    return sum(
        int(block.main_block.dyn_frozen.sum())
        for block in model.transformer.h
        if hasattr(block.main_block, "dyn_frozen")
    )


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save_ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--min_selected_epoch", type=int, default=0,
        help="Lowest epoch eligible for baseline dev selection. Formal finetuning "
        "uses 1 so a zero-shot checkpoint cannot suppress the intervention study.")
    parser.add_argument(
        "--fixed_epochs", type=int, default=None,
        help="Bypass per-arm dev selection and train exactly this many epochs. "
             "Choose the value on the full baseline, then reuse it for frozen arms.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Task-token cap (128 SST-2, 512 BoolQ, 4096 QuALITY by default).")
    parser.add_argument(
        "--scheduler", choices=("none", "select_once", "posthoc", "plateau"),
        default="none")
    parser.add_argument("--freeze_max_rate", type=float, default=0.25)
    parser.add_argument("--freeze_warmup_frac", type=float, default=0.5)
    parser.add_argument("--freeze_measure_batches", type=int, default=16)
    parser.add_argument("--freeze_plateau_z", type=float, default=1.5)
    parser.add_argument(
        "--freeze_content", choices=("auto", "own_prior", "own_prior_decomposed"),
        default="auto", help="Prior used by downstream scheduler arms. 'auto' follows "
        "the checkpoint's dense/decomposed representation.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force_flash_attention", action="store_true",
                        help="Disallow non-Flash SDPA fallbacks in matched systems runs.")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"),
        default="bfloat16")
    args = parser.parse_args()
    if args.grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    if args.epochs < 0 or (args.fixed_epochs is not None and args.fixed_epochs < 0):
        raise ValueError("epoch counts must be non-negative")
    if not 0 <= args.min_selected_epoch <= args.epochs:
        raise ValueError("min_selected_epoch must lie in [0, epochs]")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0.0 <= args.freeze_max_rate <= 1.0:
        raise ValueError("freeze_max_rate must lie in [0, 1]")
    if args.freeze_measure_batches <= 0:
        raise ValueError("freeze_measure_batches must be positive")
    if args.scheduler != "none" and args.fixed_epochs is None:
        raise ValueError(
            "scheduler arms require --fixed_epochs selected by the full baseline")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.force_flash_attention:
        if device.type != "cuda":
            raise ValueError("forced Flash attention requires CUDA")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model, config, source_checkpoint = load_model(args.ckpt, args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.state_size:
        raise ValueError("stateful checkpoints are not supported for task finetuning")
    max_length = args.max_length
    if max_length is None:
        max_length = (TASKS[args.task]["scheduler_max_length"]
                      if args.scheduler != "none" else config.block_size)
    if max_length > config.block_size:
        raise ValueError(
            f"max_length {max_length} exceeds checkpoint block size {config.block_size}")
    source_path = pathlib.Path(args.ckpt).resolve()
    source_sha256 = sha256_file(source_path)
    frozen_snapshot = capture_frozen_state(model)
    frozen_before = summarize_frozen_state(
        frozen_snapshot, config.n_layer * config.n_head)
    if args.scheduler != "none" and frozen_before["frozen_count"] != 0:
        raise ValueError("scheduler arms must start from the same unfrozen checkpoint")
    if args.scheduler != "none" and not any(
            hasattr(block.main_block, "dyn_frozen") for block in model.transformer.h):
        raise ValueError("checkpoint was not built with dynamic attention enabled")
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset_args = TASKS[args.task]["dataset"]
    raw = load_dataset(*dataset_args)
    train_indices, dev_indices = stratified_train_dev_indices(
        raw["train"], args.task, args.dev_fraction, args.seed,
        limit=args.max_train_examples)
    validation_indices = stratified_sample_indices(
        raw["validation"], args.task, args.max_eval_examples, args.seed + 2)
    train_set = PromptDataset(
        raw["train"], args.task, tokenizer, max_length,
        indices=train_indices, split_name="train")
    dev_set = PromptDataset(
        raw["train"], args.task, tokenizer, max_length,
        indices=dev_indices, split_name="selection_dev")
    validation_set = PromptDataset(
        raw["validation"], args.task, tokenizer, max_length,
        indices=validation_indices, split_name="validation")
    generator = torch.Generator().manual_seed(args.seed)
    collate_fn = lambda rows: collate(rows, tokenizer.eot_token)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=collate_fn)
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn)
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn)
    calibration_loader = None
    calibration_batches = None
    freeze_ctrl = None
    scheduler_messages = []
    scheduler_events = []
    scheduler_wall_s = 0.0
    if args.scheduler != "none":
        calibration_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + 100_003),
            collate_fn=lambda rows: collate(
                rows, tokenizer.eot_token, fixed_length=max_length))
        calibration_batches = CyclingBatches(calibration_loader, device)

        def measurement_forward(batch):
            return model.forward_hidden(batch["input_ids"])

        def scheduler_log(message):
            scheduler_messages.append(str(message))
            print(message, flush=True)

        controller_mode = "plateau" if args.scheduler == "plateau" else "select_once"
        warmup = 1.0 if args.scheduler == "posthoc" else args.freeze_warmup_frac
        freeze_content = args.freeze_content
        if freeze_content == "auto":
            freeze_content = ("own_prior_decomposed"
                              if config.rand_attn_prior_repr == "decomposed"
                              else "own_prior")
        freeze_ctrl = DynamicFreezeController(
            model, content=freeze_content, mode=controller_mode,
            check_interval=1, warmup_frac=warmup,
            max_rate=args.freeze_max_rate,
            measure_batches=args.freeze_measure_batches,
            select_signal="variance", plateau_z=args.freeze_plateau_z,
            variance_estimator="sample", seed=args.seed,
            log=scheduler_log, measurement_forward=measurement_forward)
    choice_ids = train_set.choice_ids
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    initial = evaluate(model, dev_loader, choice_ids, device, dtype)
    history.append({"epoch": 0, **initial})
    best_dev = initial if args.min_selected_epoch == 0 else None
    best_epoch = 0 if args.min_selected_epoch == 0 else None
    best_state = ({key: value.detach().cpu().clone()
                   for key, value in model.state_dict().items()}
                  if args.min_selected_epoch == 0 else None)
    optimizer.zero_grad(set_to_none=True)
    train_epochs = args.fixed_epochs if args.fixed_epochs is not None else args.epochs
    selection_mode = "fixed_epochs" if args.fixed_epochs is not None else "dev_best"
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_optimizer_updates = train_epochs * updates_per_epoch
    optimizer_update = 0
    last_scheduler_frozen_count = frozen_before["frozen_count"]
    optimizer_train_wall_s = 0.0
    optimizer_epoch_wall_s = []
    plateau_loss_observation_wall_s = 0.0

    def record_scheduler_call(kind, update, callback):
        nonlocal scheduler_wall_s, last_scheduler_frozen_count
        message_count = len(scheduler_messages)
        if kind != "plateau" and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        callback()
        # Reading a CUDA mask count synchronizes the device. Do it only after
        # the controller logs a freeze/rollback/stop, never on every plateau
        # observation (short-task updates take only tens of milliseconds).
        if len(scheduler_messages) != message_count:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            after = frozen_head_count(model)
        else:
            after = last_scheduler_frozen_count
        elapsed = time.perf_counter() - start
        scheduler_wall_s += elapsed
        if after != last_scheduler_frozen_count:
            event = {
                "kind": kind,
                "optimizer_update": update,
                "training_fraction": (
                    update / total_optimizer_updates
                    if total_optimizer_updates else 1.0),
                "frozen_before": last_scheduler_frozen_count,
                "frozen_after": after,
                "wall_s": elapsed,
            }
            scheduler_events.append(event)
            print(json.dumps({"scheduler_event": event}), flush=True)
        last_scheduler_frozen_count = after

    for epoch in range(1, train_epochs + 1):
        model.train()
        group_loss_sum = None
        group_examples = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_train_start = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            inputs, lengths = batch["input_ids"], batch["lengths"]
            targets = batch["targets"]
            inputs, lengths = inputs.to(device), lengths.to(device)
            targets = targets.to(device)
            group_start = ((step - 1) // args.grad_accum) * args.grad_accum + 1
            group_end = min(group_start + args.grad_accum - 1, len(train_loader))
            group_size = group_end - group_start + 1
            if step == group_start:
                group_loss_sum = None
                group_examples = 0
                if (args.scheduler == "select_once"
                        and optimizer_update == int(
                            args.freeze_warmup_frac * total_optimizer_updates)):
                    record_scheduler_call(
                        "select_once", optimizer_update,
                        lambda: freeze_ctrl.maybe_freeze(
                            optimizer_update, total_optimizer_updates,
                            calibration_batches, autocast(device, dtype)))
            with autocast(device, dtype):
                logits = answer_logits(model, inputs, lengths)
                task_loss = F.cross_entropy(logits, targets)
                loss = task_loss / group_size
            weighted_loss = task_loss.detach() * inputs.size(0)
            group_loss_sum = (weighted_loss if group_loss_sum is None
                              else group_loss_sum + weighted_loss)
            group_examples += inputs.size(0)
            loss.backward()
            if step == group_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.scheduler == "plateau":
                    observation_start = time.perf_counter()
                    observed_loss = float(group_loss_sum / group_examples)
                    plateau_loss_observation_wall_s += (
                        time.perf_counter() - observation_start)
                    record_scheduler_call(
                        "plateau", optimizer_update,
                        lambda: freeze_ctrl.observe_loss(
                            optimizer_update, total_optimizer_updates,
                            observed_loss,
                            calibration_batches, autocast(device, dtype)))
                optimizer_update += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_wall_s = time.perf_counter() - epoch_train_start
        optimizer_epoch_wall_s.append(epoch_wall_s)
        optimizer_train_wall_s += epoch_wall_s
        metrics = evaluate(model, dev_loader, choice_ids, device, dtype)
        history.append({"epoch": epoch, **metrics})
        print(json.dumps(history[-1]), flush=True)
        if args.fixed_epochs is not None:
            # The epoch count was selected on a different (full-baseline) run.
            # Keep the final state irrespective of this arm's dev trajectory.
            if epoch == train_epochs:
                best_dev = metrics
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
        elif epoch >= args.min_selected_epoch and (
                best_dev is None or metrics["accuracy"] > best_dev["accuracy"] or (
                    metrics["accuracy"] == best_dev["accuracy"]
                    and metrics["choice_nll"] < best_dev["choice_nll"])):
            best_dev = metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}

    if optimizer_update != total_optimizer_updates:
        raise RuntimeError(
            f"optimizer update accounting mismatch: {optimizer_update} != "
            f"{total_optimizer_updates}")

    # Model/epoch selection used only the internal training split. Restore that
    # state, then evaluate the official validation split exactly once.
    if best_state is None or best_dev is None or best_epoch is None:
        raise RuntimeError("no epoch was eligible for downstream model selection")
    model.load_state_dict(best_state)
    if args.scheduler == "none" and not frozen_state_matches(model, frozen_snapshot):
        raise RuntimeError("task finetuning changed the frozen-head mask or patterns")
    if args.scheduler == "posthoc":
        record_scheduler_call(
            "posthoc", total_optimizer_updates,
            lambda: freeze_ctrl.maybe_freeze(
                total_optimizer_updates, total_optimizer_updates,
                calibration_batches, autocast(device, dtype)))
    frozen_after_snapshot = capture_frozen_state(model)
    frozen_after = summarize_frozen_state(
        frozen_after_snapshot, config.n_layer * config.n_head)
    official_validation = evaluate(
        model, validation_loader, choice_ids, device, dtype,
        return_examples=True)
    official_examples = official_validation.pop("examples")
    print(json.dumps({
        "selected_epoch": best_epoch,
        "official_validation": official_validation,
    }), flush=True)

    dataset_fingerprints = {
        split: getattr(raw[split], "_fingerprint", None)
        for split in ("train", "validation")
    }
    selected_state_updates = best_epoch * updates_per_epoch
    selected_state_train_wall_s = sum(optimizer_epoch_wall_s[:best_epoch])
    payload = {
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_iter": source_checkpoint.get("iter_num"),
        "task": args.task,
        "task_definition": {
            "lm_eval_version": "0.4.12",
            "task_version": TASKS[args.task]["task_version"],
            "dataset_path": dataset_args[0],
            "dataset_name": dataset_args[1] if len(dataset_args) > 1 else None,
            "choices": list(TASKS[args.task]["choices"]),
            "target_delimiter": " ",
            "prompt_template": TASKS[args.task]["prompt_template"],
            "dataset_fingerprints": dataset_fingerprints,
        },
        "position_encoding": config.position_encoding,
        "max_length": max_length,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "min_selected_epoch": args.min_selected_epoch,
        "epochs_trained": train_epochs,
        "fixed_epochs": args.fixed_epochs,
        "selection_mode": selection_mode,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "flash_sdpa_forced": args.force_flash_attention,
        "wall_s_before_result_export": time.perf_counter() - process_start,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        "memory_scope": "task pipeline after model loading, including evaluation",
        "train_examples": len(train_set),
        "selection_dev_examples": len(dev_set),
        "validation_examples": len(validation_set),
        "mean_train_token_length": float(np.mean([
            len(example["tokens"]) for example in train_set.examples])),
        "mean_validation_token_length": float(np.mean([
            len(example["tokens"]) for example in validation_set.examples])),
        "dev_fraction": args.dev_fraction,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "optimizer_updates_trained": total_optimizer_updates,
        "optimizer_train_wall_s": optimizer_train_wall_s,
        "optimizer_epoch_wall_s": optimizer_epoch_wall_s,
        "mean_ms_per_optimizer_update": (
            1000.0 * optimizer_train_wall_s / total_optimizer_updates
            if total_optimizer_updates else None),
        "selected_state_optimizer_updates": selected_state_updates,
        "selected_state_train_wall_s": selected_state_train_wall_s,
        "selected_state_mean_ms_per_optimizer_update": (
            1000.0 * selected_state_train_wall_s / selected_state_updates
            if selected_state_updates else None),
        "selection_history": history,
        "selected_epoch": best_epoch,
        "selected_dev_epoch": best_epoch if args.fixed_epochs is None else None,
        "best_selection_dev": best_dev,
        "official_validation": official_validation,
        "official_validation_examples": official_examples,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "frozen_unchanged": frozen_state_matches(model, frozen_snapshot),
        "scheduler": {
            "name": args.scheduler,
            "content": freeze_content if freeze_ctrl is not None else None,
            "selection_signal": "variance" if freeze_ctrl is not None else None,
            "max_rate": args.freeze_max_rate if freeze_ctrl is not None else None,
            "warmup_fraction": (
                args.freeze_warmup_frac
                if args.scheduler == "select_once" else None),
            "measure_batches": (
                args.freeze_measure_batches if freeze_ctrl is not None else None),
            "plateau_z": args.freeze_plateau_z if args.scheduler == "plateau" else None,
            "calibration_split": "train" if freeze_ctrl is not None else None,
            "calibration_sequence_length": max_length if freeze_ctrl is not None else None,
            "calibration_batch_order_seed": (
                args.seed + 100_003 if freeze_ctrl is not None else None),
            "calibration_note": (
                "Calibration batches are right-padded to the task cap so every "
                "absolute prior covers every downstream sequence; normal task "
                "training and evaluation remain dynamically padded. Query rows "
                "after a sample's real prompt are therefore padding-derived and "
                "can bias late-query prior means; this is a screening limitation."
                if freeze_ctrl is not None else None),
            "events": scheduler_events,
            "messages": scheduler_messages,
            "controller_wall_s": scheduler_wall_s,
            "loss_observation_wall_s": plateau_loss_observation_wall_s,
            "total_observed_overhead_wall_s": (
                scheduler_wall_s + plateau_loss_observation_wall_s),
        },
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.save_ckpt:
        save_path = pathlib.Path(args.save_ckpt)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        torch.save({
            "model": final_state,
            "model_args": asdict(config),
            "source_config": source_checkpoint.get("config"),
            "source_checkpoint_sha256": source_sha256,
            "task_finetune": payload,
            "attention_layout": source_checkpoint.get("attention_layout"),
        }, save_path)


if __name__ == "__main__":
    main()

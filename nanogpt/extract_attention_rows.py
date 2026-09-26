"""Extract fixed-query causal attention logits for covariance diagnostics.

The output stores only valid causal prefixes, concatenated across query
positions. Attention probabilities can be reconstructed exactly with a
row-wise softmax, while CLR attention coordinates are simply the logits with
their valid-row mean removed.
"""

import argparse
import contextlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_attention_features import attention_modules, project_qk, sha256_file
from model import GPT, GPTConfig


DEFAULT_QUERY_POSITIONS = "63,127,255,511,767,1023"
EXPECTED_SELECTED_HEADS = 36
SCHEMA_VERSION = 2


def parse_query_positions(value):
    try:
        positions = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "query_positions must be a comma-separated list of integers") from exc
    if not positions:
        raise argparse.ArgumentTypeError("query_positions must not be empty")
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise argparse.ArgumentTypeError(
            "query_positions must be unique and strictly increasing")
    if positions[0] < 0:
        raise argparse.ArgumentTypeError("query_positions must be non-negative")
    return positions


def parse_selection(path, n_layer, n_head, checkpoint_path, dataset):
    selection_path = pathlib.Path(path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"selection JSON not found: {selection_path}")
    payload = json.loads(selection_path.read_text())
    labels = payload.get("selected_heads")
    if not isinstance(labels, list) or not labels:
        raise ValueError("selection JSON must contain a non-empty selected_heads list")
    if len(labels) != EXPECTED_SELECTED_HEADS:
        raise ValueError(
            f"expected exactly {EXPECTED_SELECTED_HEADS} selected heads, got {len(labels)}")
    if payload.get("n_frozen") is not None and int(payload["n_frozen"]) != len(labels):
        raise ValueError("selection JSON n_frozen disagrees with selected_heads")
    if payload.get("n_heads") is not None and int(payload["n_heads"]) != n_layer * n_head:
        raise ValueError("selection JSON n_heads disagrees with checkpoint architecture")
    if payload.get("dataset") is not None and payload["dataset"] != dataset:
        raise ValueError(
            f"selection dataset {payload['dataset']!r} does not match {dataset!r}")

    selection_checkpoint = payload.get("checkpoint")
    if selection_checkpoint:
        candidate = pathlib.Path(selection_checkpoint).expanduser()
        if candidate.exists() and candidate.resolve() != checkpoint_path.resolve():
            raise ValueError(
                "selection JSON was produced from a different checkpoint: "
                f"{candidate.resolve()} != {checkpoint_path.resolve()}")

    pairs = []
    seen = set()
    for label in labels:
        match = re.fullmatch(r"L(\d+)H(\d+)", str(label))
        if match is None:
            raise ValueError(f"invalid selected-head label: {label!r}")
        layer, head = map(int, match.groups())
        if not (0 <= layer < n_layer and 0 <= head < n_head):
            raise ValueError(f"selected head out of range: {label}")
        if (layer, head) in seen:
            raise ValueError(f"duplicate selected head: {label}")
        seen.add((layer, head))
        pairs.append((layer, head))
    return selection_path, payload, [str(label) for label in labels], pairs


def sample_unique_offsets(population_size, count, seed, min_separation=1):
    """Draw deterministic starts without allocating a corpus-sized candidate array."""
    if population_size <= 0:
        raise ValueError("validation data is shorter than the model context")
    if count <= 0:
        raise ValueError("num_samples must be positive")
    if count > population_size:
        raise ValueError(
            f"cannot draw {count} unique offsets from {population_size} positions")
    if min_separation <= 0:
        raise ValueError("min_separation must be positive")
    max_pack = 1 + (population_size - 1) // min_separation
    if count > max_pack:
        raise ValueError(
            f"cannot place {count} offsets at separation {min_separation} "
            f"inside {population_size} positions")
    rng = np.random.default_rng(seed)
    offsets = []
    seen = set()
    while len(offsets) < count:
        need = count - len(offsets)
        candidates = rng.integers(0, population_size, size=max(32, 2 * need), dtype=np.int64)
        for value in candidates:
            item = int(value)
            if item not in seen and all(
                    abs(item - existing) >= min_separation for existing in offsets):
                seen.add(item)
                offsets.append(item)
                if len(offsets) == count:
                    break
    return np.asarray(offsets, dtype=np.int64)


def git_head(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def model_autocast(device_type):
    if device_type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def precision_mode_for_device(device_type):
    if device_type == "cuda":
        return "model_bfloat16_autocast_qk_matmul_then_float32_storage"
    if device_type == "cpu":
        return "model_float32_qk_matmul_and_storage"
    raise ValueError(f"unsupported device type: {device_type}")


def selected_query_logits(module, normalized, local_heads, query_positions,
                          rope=None):
    """Compute selected raw logits in the caller's active model precision.

    This deliberately performs no dtype conversion. Under CUDA autocast it
    follows the model's BF16 Q/K projection and BF16 matmul path; callers cast
    only the resulting logits to float32 for storage.
    """
    q, k = project_qk(module, normalized, rope=rope)
    q_selected = q.index_select(1, local_heads).index_select(2, query_positions)
    k_selected = k.index_select(1, local_heads)
    return (q_selected @ k_selected.transpose(-2, -1)) * (
        1.0 / math.sqrt(k_selected.size(-1)))


def validate_standard_attention(modules, state_size):
    if state_size:
        raise ValueError("stateful checkpoints are not supported by the row extractor")
    for layer, module in enumerate(modules):
        frozen_state = getattr(module, "dyn_frozen", None)
        frozen = frozen_state is not None and bool(frozen_state.any())
        nonstandard = (
            module.n_pattern_heads > 0
            or module.n_random_heads > 0
            or module.attn_mask_rate > 0
            or getattr(module, "pattern_override", None) is not None
            or frozen
        )
        if nonstandard:
            raise ValueError(
                f"layer {layer} has nonstandard/frozen attention; raw QK logits "
                "would not describe the attention used by the checkpoint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="fineweb_edu")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--query_positions", type=parse_query_positions,
        default=parse_query_positions(DEFAULT_QUERY_POSITIONS),
    )
    parser.add_argument("--data_seed", type=int, default=20260716)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("num_samples must be positive")
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.data_seed < 0:
        parser.error("data_seed must be non-negative")

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    checkpoint_path = pathlib.Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    data_path = repo_root / "data" / args.dataset / f"{args.split}.bin"
    if not data_path.is_file():
        raise FileNotFoundError(f"{args.split} data not found: {data_path}")
    out_path = pathlib.Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device type: {device.type}")
    torch.manual_seed(args.data_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.data_seed)
    precision_mode = precision_mode_for_device(device.type)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_args" not in checkpoint or "model" not in checkpoint:
        raise ValueError("checkpoint must contain model_args and model")
    checkpoint_iter = checkpoint.get("iter_num")
    if checkpoint_iter is not None:
        checkpoint_iter = int(
            checkpoint_iter.item() if torch.is_tensor(checkpoint_iter) else checkpoint_iter)
    model_args = dict(checkpoint["model_args"])
    if "rand_mask_rate" in model_args:
        model_args["rand_proj_mask_rate"] = model_args.pop("rand_mask_rate")
    model = GPT(GPTConfig(**model_args))
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model.to(device).eval()

    modules = attention_modules(model)
    validate_standard_attention(modules, model.config.state_size)
    selection_path, selection_payload, head_labels, selected_pairs = parse_selection(
        args.selection_json, model.config.n_layer, model.config.n_head,
        checkpoint_path, args.dataset,
    )

    seq_len = model.config.block_size
    positions = list(args.query_positions)
    if positions[-1] >= seq_len:
        raise ValueError(
            f"query position {positions[-1]} is outside block size {seq_len}")
    query_slices = []
    cursor = 0
    for position in positions:
        end = cursor + position + 1
        query_slices.append((cursor, end))
        cursor = end
    concatenated_width = cursor

    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    population_size = len(data) - seq_len + 1
    offsets = sample_unique_offsets(
        population_size, args.num_samples, args.data_seed,
        min_separation=seq_len,
    )
    selected_by_layer = {layer: [] for layer in range(model.config.n_layer)}
    for output_index, (layer, head) in enumerate(selected_pairs):
        selected_by_layer[layer].append((output_index, head))

    raw_temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{out_path.name}.", suffix=".raw.npy",
        dir=out_path.parent, delete=False,
    )
    raw_temp_path = pathlib.Path(raw_temp_handle.name)
    raw_temp_handle.close()
    output_temp_path = None
    raw_logits = None
    try:
        raw_logits = np.lib.format.open_memmap(
            raw_temp_path, mode="w+", dtype=np.float32,
            shape=(args.num_samples, len(selected_pairs), concatenated_width),
        )
        positions_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        with torch.inference_mode():
            for sample_start in range(0, args.num_samples, args.batch_size):
                sample_end = min(sample_start + args.batch_size, args.num_samples)
                batch_offsets = offsets[sample_start:sample_end]
                tokens = torch.stack([
                    torch.from_numpy(data[offset:offset + seq_len].astype(np.int64))
                    for offset in batch_offsets
                ]).to(device)
                batch_raw = np.full(
                    (len(batch_offsets), len(selected_pairs), concatenated_width),
                    np.nan, dtype=np.float32,
                )

                with model_autocast(device.type):
                    hidden, rope = model.prepare_inputs(tokens)

                for layer, (block, module) in enumerate(zip(model.transformer.h, modules)):
                    entries = selected_by_layer[layer]
                    if entries:
                        output_indices = [item[0] for item in entries]
                        local_heads = torch.tensor(
                            [item[1] for item in entries], dtype=torch.long, device=device)
                        # Match the model path exactly: LN, Q/K projections and
                        # QK^T all execute in its active autocast precision.
                        with model_autocast(device.type):
                            normalized = block.ln_1(hidden)
                            logits = selected_query_logits(
                                module, normalized, local_heads, positions_tensor,
                                rope=rope)
                        expected_dtype = (
                            torch.bfloat16 if device.type == "cuda" else torch.float32)
                        if logits.dtype != expected_dtype:
                            raise RuntimeError(
                                f"expected {expected_dtype} model-path logits, got "
                                f"{logits.dtype}")
                        if not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError(
                                f"non-finite raw logits in layer {layer}")
                        logits_cpu = logits.float().cpu().numpy()
                        for query_index, (start, end) in enumerate(query_slices):
                            valid_length = positions[query_index] + 1
                            batch_raw[:, output_indices, start:end] = \
                                logits_cpu[:, :, query_index, :valid_length]

                    if layer + 1 < model.config.n_layer:
                        with model_autocast(device.type):
                            hidden = block(hidden, rope=rope)

                if not np.isfinite(batch_raw).all():
                    raise RuntimeError(
                        f"one or more selected heads were not populated for samples "
                        f"{sample_start}:{sample_end}")
                raw_logits[sample_start:sample_end] = batch_raw
                batch_number = sample_start // args.batch_size + 1
                if sample_end == args.num_samples or batch_number % 10 == 0:
                    raw_logits.flush()
                print(
                    f"samples {sample_end}/{args.num_samples}", flush=True)

        checkpoint_digest = sha256_file(checkpoint_path)
        selection_digest = sha256_file(selection_path)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "precision_mode": precision_mode,
            "layout": "concatenated_valid_causal_rows",
            "stored_quantity": "scaled_pre_mask_qk_logits",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_iter": checkpoint_iter,
            "selection_json": str(selection_path),
            "selection_json_sha256": selection_digest,
            "selection_source_commit": selection_payload.get("source_commit"),
            "dataset": args.dataset,
            "split": args.split,
            "data_file": str(data_path.resolve()),
            "data_seed": args.data_seed,
            "sampling_without_replacement": True,
            "non_overlapping_windows": True,
            "offset_min_separation": seq_len,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "seq_len": seq_len,
            "selected_head_count": len(selected_pairs),
            "head_labels": head_labels,
            "head_order": "selection_json.selected_heads",
            "query_positions": positions,
            "query_slices": query_slices,
            "query_starts": [start for start, _ in query_slices],
            "query_lengths": [end - start for start, end in query_slices],
            "concatenated_width": concatenated_width,
            "raw_logits_shape": [
                args.num_samples, len(selected_pairs), concatenated_width],
            "raw_logits_dtype": "float32",
            "future_keys_stored": False,
            "valid_key_rule": "key_position <= query_position",
            "logit_scale": "1/sqrt(head_dim)",
            "qk_recompute": "exact model autocast path; cast to float32 only for storage",
            "git_sha": git_head(repo_root),
            "source_commit": os.environ.get("SOURCE_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        }

        output_temp_handle = tempfile.NamedTemporaryFile(
            prefix=f".{out_path.name}.", suffix=".npz",
            dir=out_path.parent, delete=False,
        )
        output_temp_path = pathlib.Path(output_temp_handle.name)
        output_temp_handle.close()
        np.savez_compressed(
            output_temp_path,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
            precision_mode=np.asarray(precision_mode),
            raw_logits=raw_logits,
            query_slices=np.asarray(query_slices, dtype=np.int64),
            query_positions=np.asarray(positions, dtype=np.int64),
            query_starts=np.asarray([start for start, _ in query_slices], dtype=np.int64),
            query_lengths=np.asarray([end - start for start, end in query_slices], dtype=np.int64),
            head_labels=np.asarray(head_labels),
            selected_head_indices=np.asarray(selected_pairs, dtype=np.int64),
            offsets=offsets,
            metadata=np.asarray(json.dumps(metadata)),
        )
        with np.load(output_temp_path, allow_pickle=False) as check:
            expected_shape = (args.num_samples, len(selected_pairs), concatenated_width)
            if check["raw_logits"].shape != expected_shape:
                raise RuntimeError("saved raw-logit shape failed verification")
            if int(check["schema_version"].item()) != SCHEMA_VERSION:
                raise RuntimeError("saved schema version failed verification")
            if str(check["precision_mode"].item()) != precision_mode:
                raise RuntimeError("saved precision mode failed verification")
            if not np.array_equal(check["query_positions"], np.asarray(positions)):
                raise RuntimeError("saved query positions failed verification")
            if not np.array_equal(
                    check["query_lengths"], np.asarray(positions, dtype=np.int64) + 1):
                raise RuntimeError("saved query lengths failed verification")
            if not np.array_equal(check["offsets"], offsets):
                raise RuntimeError("saved offsets failed verification")
        os.replace(output_temp_path, out_path)
        output_temp_path = None
        print(
            f"saved {out_path} raw_logits="
            f"({args.num_samples}, {len(selected_pairs)}, {concatenated_width})",
            flush=True,
        )
    finally:
        if raw_logits is not None:
            raw_logits.flush()
            del raw_logits
        raw_temp_path.unlink(missing_ok=True)
        if output_temp_path is not None:
            output_temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

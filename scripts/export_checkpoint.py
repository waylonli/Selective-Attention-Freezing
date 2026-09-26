"""Export an evaluation checkpoint, preserving patterns, layouts and precision."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanogpt.model import GPTConfig


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluation_payload(source):
    args = source.get("model_args", source.get("model_config"))
    if args is None or "model" not in source:
        raise ValueError("Not a supported SAF training or evaluation checkpoint")
    config = dict(source.get("config") or {}) | dict(args)
    config = {k: v for k, v in config.items() if k in GPTConfig.__dataclass_fields__}
    if config.get("rand_attn_prior_path"):
        raise ValueError("External prior dependency: embed and validate the prior before exporting")
    layout = source.get("attention_layout")
    if layout is None and source.get("intervention"):
        intervention = source["intervention"]
        mode = intervention.get("physical_mode", intervention["mode"])
        mode = {"prune_gate_taylor": "prune", "random_mean": "mean"}.get(mode, mode)
        layout = {"mode": mode, "heads": intervention["heads"]}
    state = {key.removeprefix("_orig_mod."): value.detach().cpu()
             for key, value in source["model"].items()}
    payload = {"format_version": 1, "model": state,
               "model_args": asdict(GPTConfig(**config)), "attention_layout": layout}
    for key in ("iter_num", "step", "task", "updates", "source_checkpoint_sha256"):
        if key in source:
            payload[key] = source[key]
    # Do not export full task records: they contain dataset examples and paths.
    task = source.get("task_finetune")
    if task:
        keys = ("task", "seed", "max_length", "batch_size", "grad_accum", "selected_epoch",
                "learning_rate", "weight_decay", "dev_fraction", "data_split_sha256")
        payload["task_finetune"] = {k: task[k] for k in keys if k in task}
    return payload


def export(source, out, expected_sha256=None):
    if out.exists():
        raise FileExistsError(out)
    source_hash = sha256(source)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    original = torch.load(source, map_location="cpu", weights_only=False)
    payload = evaluation_payload(original)
    del original
    out.mkdir(parents=True)
    target = out / "model.pt"
    torch.save(payload, target)
    # Strictly reload, including dynamic buffers and physically pruned projections.
    from nanogpt.eval_nanogpt_logprobs import load_model
    model, _, restored = load_model(target, "cpu")
    for key, value in payload["model"].items():
        if not torch.equal(value, restored["model"][key]):
            raise RuntimeError(f"Export changed tensor {key}")
    metadata = {"source_sha256": source_hash, "export_sha256": sha256(target),
                "source_bytes": source.stat().st_size, "export_bytes": target.stat().st_size,
                "tensor_count": len(payload["model"]), "strict_load_passed": True,
                "optimiser_included": False, "dtype_conversion": False}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(payload["model_args"], indent=2) + "\n")
    if payload["attention_layout"] is not None:
        (out / "attention_layout.json").write_text(json.dumps(payload["attention_layout"], indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--trust-source", action="store_true",
                        help="Acknowledge that the original training pickle is trusted")
    args = parser.parse_args()
    if not args.trust_source:
        parser.error("Only load your own trusted training checkpoints; pass --trust-source")
    print(json.dumps(export(args.source, args.out, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()

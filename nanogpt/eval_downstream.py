"""Run standard EleutherAI downstream tasks on a nanoGPT checkpoint."""

import argparse
import hashlib
import json
import os
import pathlib

from lm_eval import simple_evaluate, utils

from lm_eval_adapter import NanoGPTLM


def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_limit(value):
    if value is None:
        return None
    return float(value) if any(char in value for char in ".eE") else int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tasks", default="hellaswag,piqa,arc_easy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"),
        default="bfloat16")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=parse_limit, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no_log_samples", action="store_true")
    args = parser.parse_args()

    model = NanoGPTLM(
        args.ckpt, device=args.device, batch_size=args.batch_size,
        dtype=args.dtype)
    checkpoint_path = pathlib.Path(args.ckpt).resolve()
    results = simple_evaluate(
        model=model,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=not args.no_log_samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    frozen_heads = [
        f"L{layer}H{head}"
        for layer, block in enumerate(model.model.transformer.h)
        if hasattr(block.main_block, "dyn_frozen")
        for head in block.main_block.dyn_frozen.nonzero(
            as_tuple=True)[0].tolist()
    ]
    results.setdefault("nanogpt", {}).update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_iter": model.checkpoint.get("iter_num"),
        "position_encoding": model.config.position_encoding,
        "rope_base": model.config.rope_base,
        "frozen_heads": frozen_heads,
        "frozen_rate": len(frozen_heads) /
        (model.config.n_layer * model.config.n_head),
        "lm_eval_version": __import__("lm_eval").__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        results, indent=2, default=utils.handle_non_serializable) + "\n")
    print(json.dumps(results.get("results", {}), indent=2,
                     default=utils.handle_non_serializable))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

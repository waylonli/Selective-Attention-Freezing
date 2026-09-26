"""Stream compact per-sequence attention features for UMAP/mode analysis."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from attention_features import qk_attention_features, stratified_query_positions
from model import GPT, GPTConfig


def attention_modules(model):
    modules = []
    for block in model.transformer.h:
        module = block.main_block
        if hasattr(module, "block"):
            module = module.block
        modules.append(module)
    return modules


def project_qk(module, x, rope=None):
    if module.n_random_heads > 0 and module.n_learned_heads > 0:
        q = torch.cat([module.q_proj_rand(x), module.q_proj(x)], dim=-1)
        k = torch.cat([module.k_proj_rand(x), module.k_proj(x)], dim=-1)
    else:
        q, k = module.q_proj(x), module.k_proj(x)
    B, T, _ = q.shape
    q = q.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
    k = k.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
    return module._apply_rope(q, k, rope)


def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="fineweb_edu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_queries", type=int, default=64)
    parser.add_argument("--query_chunk", type=int, default=16)
    parser.add_argument("--query_bins", type=int, default=8)
    parser.add_argument("--distance_bins", type=int, default=12)
    parser.add_argument("--local_windows", default="4,16,64")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model_args = dict(checkpoint["model_args"])
    if "rand_mask_rate" in model_args:
        model_args["rand_proj_mask_rate"] = model_args.pop("rand_mask_rate")
    model = GPT(GPTConfig(**model_args))
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state)
    model.to(args.device).eval()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / args.dataset / f"{args.split}.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    T = model.config.block_size
    positions = stratified_query_positions(T, args.n_queries)
    windows = tuple(int(x) for x in args.local_windows.split(",") if x)
    modules = attention_modules(model)
    for li, module in enumerate(modules):
        frozen = bool(getattr(module, "dyn_frozen", torch.zeros(1)).any())
        if (module.n_pattern_heads > 0 or module.n_random_heads > 0
                or module.attn_mask_rate > 0 or frozen):
            raise ValueError(
                f"layer {li} has nonstandard/frozen attention; QK features would "
                "not describe the attention used by the checkpoint")
    all_batches = []
    all_query_batches = []
    all_offsets = []
    feature_names = None
    query_feature_names = None
    offset_generator = torch.Generator().manual_seed(args.seed)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device_type == "cuda" else \
        torch.autocast("cpu", enabled=False)
    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            offsets = torch.randint(
                len(data) - T - 1, (args.batch_size,), generator=offset_generator)
            x = torch.stack([
                torch.from_numpy(data[i:i + T].astype(np.int64)) for i in offsets.tolist()
            ]).to(args.device)
            with amp:
                hidden, rope = model.prepare_inputs(x)
                layer_features = []
                layer_query_features = []
                for block, module in zip(model.transformer.h, modules):
                    q, k = project_qk(module, block.ln_1(hidden), rope=rope)
                    features, names, query_features, query_names = qk_attention_features(
                        q, k, query_positions=positions,
                        n_query_bins=args.query_bins,
                        n_distance_bins=args.distance_bins,
                        local_windows=windows,
                        query_chunk=args.query_chunk,
                        return_per_query=True,
                    )
                    layer_features.append(features.cpu())
                    layer_query_features.append(query_features.cpu())
                    feature_names = names
                    query_feature_names = query_names
                    hidden = block(hidden, rope=rope)
            all_batches.append(torch.stack(layer_features, dim=1))  # (B,L,H,F)
            all_query_batches.append(
                torch.stack(layer_query_features, dim=1))  # (B,L,H,Q,Fq)
            all_offsets.extend(int(i) for i in offsets)
            print(f"batch {batch_idx + 1}/{args.num_batches}", flush=True)

    features = torch.cat(all_batches, dim=0).numpy().astype(np.float32)
    query_features = torch.cat(all_query_batches, dim=0).numpy().astype(np.float16)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = None
    metadata = {
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "seq_len": T,
        "query_positions": positions.tolist(),
        "query_bins": args.query_bins,
        "distance_bins": args.distance_bins,
        "local_windows": windows,
        "num_samples": int(features.shape[0]),
        "git_sha": git_sha,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=features,
        query_features=query_features,
        feature_names=np.asarray(feature_names),
        query_feature_names=np.asarray(query_feature_names),
        offsets=np.asarray(all_offsets, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"saved {out} features={features.shape}")


if __name__ == "__main__":
    main()

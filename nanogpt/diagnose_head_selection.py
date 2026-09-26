#!/usr/bin/env python3
"""Phase 1E: score head-selection signals against paired one-head labels.

The primary label replaces one head by its exact post-softmax mean attention
and measures the paired held-out NLL change. This makes all 144 heads
comparable without conflating freezability with compact-fit error. The script
also fits the deployable alpha+rho representation, reports its KL gate and
compact one-head label where accepted, and computes two output-aware signals:
head-contribution redundancy after W_O and freeze-induced block influence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_freeze import DynamicFreezeController  # noqa: E402
from eval_nanogpt_logprobs import load_model  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attention_modules(model):
    modules = []
    for block in model.transformer.h:
        module = block.main_block
        if hasattr(module, "block"):
            module = module.block
        modules.append(module)
    return modules


def amp_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_windows(
    data: np.memmap,
    seq_len: int,
    *,
    start_fraction: float,
    end_fraction: float,
    counts: tuple[int, ...],
    seed: int,
) -> list[list[int]]:
    """Return disjoint, non-overlapping window starts for each diagnostic role."""
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError("invalid validation region")
    lo = int(len(data) * start_fraction)
    hi = int(len(data) * end_fraction)
    first = (lo + seq_len - 1) // seq_len * seq_len
    starts = np.arange(first, hi - seq_len, seq_len, dtype=np.int64)
    total = sum(counts)
    if len(starts) < total:
        raise ValueError(f"need {total} windows, only {len(starts)} available")
    rng = np.random.default_rng(seed)
    selected = starts[rng.permutation(len(starts))[:total]].tolist()
    groups, cursor = [], 0
    for count in counts:
        groups.append([int(x) for x in selected[cursor:cursor + count]])
        cursor += count
    if len(set(selected)) != len(selected):
        raise AssertionError("diagnostic windows overlap")
    return groups


def batches_from_starts(
    data: np.memmap,
    starts: list[int],
    seq_len: int,
    batch_size: int,
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(starts) % batch_size:
        raise ValueError("window count must be divisible by batch_size")
    batches = []
    for begin in range(0, len(starts), batch_size):
        windows = [
            torch.from_numpy(
                np.asarray(data[s:s + seq_len + 1], dtype=np.int64).copy()
            )
            for s in starts[begin:begin + batch_size]
        ]
        tokens = torch.stack(windows).to(device)
        batches.append((tokens[:, :-1], tokens[:, 1:]))
    return batches


@torch.no_grad()
def batch_losses(model, batches, device: str, dtype: torch.dtype) -> list[float]:
    losses = []
    for inputs, targets in batches:
        with amp_context(device, dtype):
            _, loss = model(inputs, targets)
        losses.append(float(loss))
    return losses


def paired_label(
    counterfactual: list[float], baseline: list[float]
) -> dict[str, object]:
    delta = np.asarray(counterfactual, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {
        "delta_nll": mean,
        "delta_nll_se": se,
        "delta_ppl_pct": 100.0 * math.expm1(mean),
        "batch_deltas": delta.tolist(),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_rankings(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = np.asarray([bool(row["compact_accepted"]) for row in rows])
    labels = np.asarray([float(row["dense_mean_delta_nll"]) for row in rows])
    features = {
        "variance": np.asarray([float(row["attention_variance"]) for row in rows]),
        "redundancy": -np.asarray(
            [float(row["redundancy_max_cosine"]) for row in rows]
        ),
        "block_influence_cos": np.asarray(
            [float(row["block_influence_cosine"]) for row in rows]
        ),
        "block_influence_l2": np.asarray(
            [float(row["block_influence_relative_l2"]) for row in rows]
        ),
        "contribution_norm": np.asarray(
            [float(row["contribution_rms"]) for row in rows]
        ),
    }
    result: dict[str, object] = {
        "eligible_heads": int(accepted.sum()),
        "ineligible_heads": int((~accepted).sum()),
        "feature_results": {},
    }
    eligible_labels = labels[accepted]
    for name, risk in features.items():
        feature_result: dict[str, object] = {
            "spearman_risk_vs_dense_delta_nll": spearman(risk[accepted], eligible_labels),
            "rates": {},
        }
        eligible_indices = np.flatnonzero(accepted)
        ranked = eligible_indices[np.argsort(risk[accepted], kind="mergesort")]
        for rate in (0.25, 0.50):
            count = int(144 * rate)
            chosen = ranked[:count]
            feature_result["rates"][f"{int(rate * 100)}"] = {
                "count": count,
                "mean_dense_delta_nll": float(labels[chosen].mean()),
                "mean_dense_delta_ppl_pct": float(
                    100.0 * np.expm1(labels[chosen]).mean()
                ),
                "heads": [str(rows[i]["label"]) for i in chosen],
            }
        result["feature_results"][name] = feature_result
    variance = result["feature_results"]["variance"]["rates"]
    promotions = []
    for name, feature_result in result["feature_results"].items():
        if name == "variance":
            continue
        rates = feature_result["rates"]
        if all(
            rates[key]["mean_dense_delta_nll"]
            < variance[key]["mean_dense_delta_nll"]
            for key in ("25", "50")
        ):
            promotions.append(name)
    result["diagnostic_promotion_candidates"] = promotions
    result["promotion_requires_full_training"] = True
    return result


@torch.no_grad()
def contribution_features(
    model,
    modules,
    means,
    batches,
    *,
    n_positions: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Output-aware features on exact mean-prior counterfactuals.

    Positions are sampled evenly from the later half of the sequence. This is
    only a feature estimator; the primary labels below use every token.
    """
    n_layer, n_head = len(modules), modules[0].n_head
    total = n_layer * n_head
    gram = np.zeros((n_layer, n_head, n_head), dtype=np.float64)
    norm = np.zeros((n_layer, n_head), dtype=np.float64)
    contribution_elements = np.zeros(n_layer, dtype=np.int64)
    influence_cos = np.zeros(total, dtype=np.float64)
    influence_l2 = np.zeros(total, dtype=np.float64)
    observations = np.zeros(total, dtype=np.int64)
    batch_redundancy = np.zeros((len(batches), total), dtype=np.float64)
    batch_contribution = np.zeros((len(batches), total), dtype=np.float64)
    batch_influence_cos = np.zeros((len(batches), total), dtype=np.float64)
    batch_influence_l2 = np.zeros((len(batches), total), dtype=np.float64)

    for batch_index, (inputs, _) in enumerate(batches):
        with amp_context(device, dtype):
            hidden, rope = model.prepare_inputs(inputs)
            seq_len = hidden.size(1)
            positions = torch.linspace(
                seq_len // 2, seq_len - 1,
                min(n_positions, seq_len - seq_len // 2),
                device=hidden.device,
            ).round().long().unique()
            keys = torch.arange(seq_len, device=hidden.device)
            causal = keys.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
            for layer, (block, module) in enumerate(
                zip(model.transformer.h, modules)
            ):
                normalized = block.ln_1(hidden)
                batch, _, channels = normalized.shape
                head_dim = module.head_dim
                q = module.q_proj(normalized).view(
                    batch, seq_len, n_head, head_dim
                ).transpose(1, 2)
                k = module.k_proj(normalized).view(
                    batch, seq_len, n_head, head_dim
                ).transpose(1, 2)
                v = module.v_proj(normalized).view(
                    batch, seq_len, n_head, head_dim
                ).transpose(1, 2)
                q, k = module._apply_rope(q, k, rope)
                scores = torch.matmul(
                    q.index_select(2, positions), k.transpose(-2, -1)
                ) / math.sqrt(head_dim)
                attention = torch.softmax(
                    scores.float().masked_fill(~causal, float("-inf")), dim=-1
                ).to(v.dtype)
                head_values = torch.matmul(attention, v)

                mean_rows = means[layer].index_select(
                    1, positions.cpu()
                ).to(device=device, dtype=v.dtype, non_blocking=True)
                prior_values = torch.einsum("hpk,bhkd->bhpd", mean_rows, v)
                weight = module.c_proj.weight.view(
                    channels, n_head, head_dim
                ).permute(1, 0, 2)
                contributions = torch.einsum(
                    "bhpd,hcd->bhpc", head_values, weight
                )
                prior_contributions = torch.einsum(
                    "bhpd,hcd->bhpc", prior_values, weight
                )
                delta = prior_contributions - contributions
                flat = contributions.float().permute(1, 0, 2, 3).reshape(
                    n_head, -1
                )
                batch_gram = (flat @ flat.T).double().cpu().numpy()
                batch_norm = flat.square().sum(1).double().cpu().numpy()
                gram[layer] += batch_gram
                norm[layer] += batch_norm
                contribution_elements[layer] += flat.shape[1]
                batch_similarity = batch_gram / np.sqrt(
                    np.outer(batch_norm, batch_norm)
                ).clip(min=1e-12)
                np.fill_diagonal(batch_similarity, -np.inf)
                start = layer * n_head
                batch_redundancy[
                    batch_index, start:start + n_head
                ] = batch_similarity.max(axis=1)
                batch_contribution[
                    batch_index, start:start + n_head
                ] = np.sqrt(batch_norm / max(1, flat.shape[1]))

                joined = head_values.transpose(1, 2).reshape(
                    batch, len(positions), channels
                )
                attention_output = F.linear(
                    joined, module.c_proj.weight, module.c_proj.bias
                )
                residual = hidden.index_select(1, positions) + attention_output
                counterfactual = residual.unsqueeze(1) + delta
                normal = residual.unsqueeze(1).expand_as(counterfactual)
                cos = 1.0 - F.cosine_similarity(
                    normal.float(), counterfactual.float(), dim=-1
                )
                rel_l2 = delta.float().norm(dim=-1) / attention_output.float().norm(
                    dim=-1
                ).unsqueeze(1).clamp_min(1e-8)
                cos_values = cos.mean(dim=(0, 2)).double().cpu().numpy()
                l2_values = rel_l2.mean(dim=(0, 2)).double().cpu().numpy()
                influence_cos[start:start + n_head] += cos_values
                influence_l2[start:start + n_head] += l2_values
                batch_influence_cos[
                    batch_index, start:start + n_head
                ] = cos_values
                batch_influence_l2[
                    batch_index, start:start + n_head
                ] = l2_values
                observations[start:start + n_head] += 1
                hidden = block(hidden, rope=rope)
        print(f"[features] batch {batch_index + 1}/{len(batches)}", flush=True)

    similarity = np.zeros_like(gram)
    redundancy_max = np.zeros((n_layer, n_head), dtype=np.float64)
    redundancy_top3 = np.zeros((n_layer, n_head), dtype=np.float64)
    for layer in range(n_layer):
        denom = np.sqrt(np.outer(norm[layer], norm[layer])).clip(min=1e-12)
        similarity[layer] = gram[layer] / denom
        np.fill_diagonal(similarity[layer], -np.inf)
        redundancy_max[layer] = similarity[layer].max(axis=1)
        redundancy_top3[layer] = np.sort(similarity[layer], axis=1)[:, -3:].mean(1)
    contribution_rms = np.sqrt(
        norm / contribution_elements[:, None].clip(min=1)
    )
    return {
        "redundancy_max": redundancy_max.reshape(-1),
        "redundancy_top3": redundancy_top3.reshape(-1),
        "contribution_rms": contribution_rms.reshape(-1),
        "block_influence_cos": influence_cos / observations.clip(min=1),
        "block_influence_l2": influence_l2 / observations.clip(min=1),
        "similarity": similarity,
        "batch_redundancy_max": batch_redundancy,
        "batch_contribution_rms": batch_contribution,
        "batch_block_influence_cos": batch_influence_cos,
        "batch_block_influence_l2": batch_influence_l2,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dataset", default="fineweb_edu")
    parser.add_argument("--split", default="val")
    parser.add_argument("--region-start-fraction", type=float, default=0.0)
    parser.add_argument("--region-end-fraction", type=float, default=0.5)
    parser.add_argument("--fit-batches", type=int, default=16)
    parser.add_argument("--feature-batches", type=int, default=4)
    parser.add_argument("--label-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--feature-positions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        default="bfloat16")
    parser.add_argument("--decomp-fit-steps", type=int, default=400)
    parser.add_argument("--decomp-max-kl", type=float, default=0.2)
    parser.add_argument("--decomp-fit-backend", choices=("fft", "dense"),
                        default="fft")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model, config, checkpoint_payload = load_model(str(args.ckpt), args.device)
    del checkpoint_payload
    modules = attention_modules(model)
    if len(modules) != config.n_layer or not all(
        getattr(module, "dynamic", False) for module in modules
    ):
        raise RuntimeError("Phase 1E requires dynamic-capable standard attention")
    if any(bool(module.dyn_frozen.any()) for module in modules):
        raise RuntimeError("Phase 1E source checkpoint must be unfrozen")
    if any(module.prior_repr != "decomposed" for module in modules):
        raise RuntimeError("Phase 1E compact labels require decomposed buffers")

    repo = Path(__file__).resolve().parents[1]
    data_path = repo / "data" / args.dataset / f"{args.split}.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    counts = (
        args.fit_batches * args.batch_size,
        args.feature_batches * args.batch_size,
        args.label_batches * args.batch_size,
    )
    fit_starts, feature_starts, label_starts = make_windows(
        data, config.block_size,
        start_fraction=args.region_start_fraction,
        end_fraction=args.region_end_fraction,
        counts=counts,
        seed=args.seed,
    )
    fit_batches = batches_from_starts(
        data, fit_starts, config.block_size, args.batch_size, args.device
    )
    feature_batches = batches_from_starts(
        data, feature_starts, config.block_size, args.batch_size, args.device
    )
    label_batches = batches_from_starts(
        data, label_starts, config.block_size, args.batch_size, args.device
    )

    fit_iter = iter(fit_batches)

    def get_fit_batch(_split):
        return next(fit_iter)

    controller = DynamicFreezeController(
        model,
        content="own_prior_decomposed",
        mode="select_once",
        max_rate=1.0,
        measure_batches=len(fit_batches),
        variance_estimator="sample",
        decomp_max_kl=args.decomp_max_kl,
        decomp_fit_steps=args.decomp_fit_steps,
        decomp_fit_backend=args.decomp_fit_backend,
        seed=args.seed,
    )
    controller._profile_selector_phases = True
    print("[capture] measuring attention means and variance", flush=True)
    with torch.no_grad():
        variance, means = controller._measure(
            get_fit_batch, amp_context(args.device, dtype)
        )
    total_heads = config.n_layer * config.n_head
    pairs = [
        (layer, head)
        for layer in range(config.n_layer)
        for head in range(config.n_head)
    ]
    causal = torch.tril(torch.ones(config.block_size, config.block_size))
    print("[fit] fitting compact alpha+rho priors for all heads", flush=True)
    accepted_pairs = set(
        controller._freeze_batch(pairs, means, config.block_size, causal)
    )
    fit_kl = {
        (layer, head): float(controller.last_decomp_kl[(layer, head)])
        for layer, head in pairs
    }
    for module in modules:
        module.unfreeze_heads(range(config.n_head))

    features = contribution_features(
        model, modules, means, feature_batches,
        n_positions=args.feature_positions,
        device=args.device,
        dtype=dtype,
    )

    print("[labels] computing paired unfrozen baseline", flush=True)
    baseline_losses = batch_losses(model, label_batches, args.device, dtype)
    dense_labels: dict[tuple[int, int], dict[str, object]] = {}
    compact_labels: dict[tuple[int, int], dict[str, object]] = {}
    for layer, module in enumerate(modules):
        for head in range(config.n_head):
            pair = (layer, head)
            pattern = means[layer][head].to(args.device, non_blocking=True)
            module.freeze_heads([head], pattern.unsqueeze(0))
            counter = batch_losses(model, label_batches, args.device, dtype)
            module.unfreeze_heads([head])
            dense_labels[pair] = paired_label(counter, baseline_losses)
            del pattern

            if pair in accepted_pairs:
                # These slices alias the module buffers written by
                # freeze_heads_decomposed.  Clone them before reinstalling one
                # head so PyTorch does not reject the overlapping copy.
                module.freeze_heads_decomposed(
                    [head],
                    module.dyn_alpha[head:head + 1].clone(),
                    module.dyn_rho[head:head + 1].clone(),
                    z=module.dyn_z[head:head + 1].clone(),
                )
                counter = batch_losses(model, label_batches, args.device, dtype)
                module.unfreeze_heads([head])
                compact_labels[pair] = paired_label(counter, baseline_losses)
            print(
                f"[labels] L{layer}H{head} dense={dense_labels[pair]['delta_nll']:+.6f} "
                f"compact={'accepted' if pair in accepted_pairs else 'KL-rejected'}",
                flush=True,
            )
        (args.out_dir / "progress.json").write_text(json.dumps({
            "completed_layers": layer + 1,
            "dense_labels": len(dense_labels),
            "compact_labels": len(compact_labels),
        }, indent=2) + "\n")

    rows: list[dict[str, object]] = []
    for layer, head in pairs:
        flat = layer * config.n_head + head
        dense = dense_labels[(layer, head)]
        compact = compact_labels.get((layer, head))
        rows.append({
            "label": f"L{layer}H{head}",
            "layer": layer,
            "head": head,
            "attention_variance": float(variance[layer, head]),
            "compact_fit_kl": fit_kl[(layer, head)],
            "compact_accepted": (layer, head) in accepted_pairs,
            "redundancy_max_cosine": float(features["redundancy_max"][flat]),
            "redundancy_top3_cosine": float(features["redundancy_top3"][flat]),
            "contribution_rms": float(features["contribution_rms"][flat]),
            "block_influence_cosine": float(features["block_influence_cos"][flat]),
            "block_influence_relative_l2": float(features["block_influence_l2"][flat]),
            "dense_mean_delta_nll": dense["delta_nll"],
            "dense_mean_delta_nll_se": dense["delta_nll_se"],
            "dense_mean_delta_ppl_pct": dense["delta_ppl_pct"],
            "compact_mean_delta_nll": (
                compact["delta_nll"] if compact is not None else None
            ),
            "compact_mean_delta_nll_se": (
                compact["delta_nll_se"] if compact is not None else None
            ),
            "compact_mean_delta_ppl_pct": (
                compact["delta_ppl_pct"] if compact is not None else None
            ),
        })

    ranking = summarize_rankings(rows)
    dense_batch_deltas = np.asarray([
        dense_labels[(layer, head)]["batch_deltas"] for layer, head in pairs
    ], dtype=np.float64)
    aggregate_dense_labels = np.asarray([
        dense_labels[(layer, head)]["delta_nll"] for layer, head in pairs
    ], dtype=np.float64)
    ranking["label_batch_rank_stability"] = [
        spearman(dense_batch_deltas[:, batch], aggregate_dense_labels)
        for batch in range(dense_batch_deltas.shape[1])
    ]
    aggregate_features = {
        "redundancy": -features["redundancy_max"],
        "block_influence_cos": features["block_influence_cos"],
        "block_influence_l2": features["block_influence_l2"],
        "contribution_norm": features["contribution_rms"],
    }
    batch_features = {
        "redundancy": -features["batch_redundancy_max"],
        "block_influence_cos": features["batch_block_influence_cos"],
        "block_influence_l2": features["batch_block_influence_l2"],
        "contribution_norm": features["batch_contribution_rms"],
    }
    ranking["feature_batch_rank_stability"] = {
        name: [spearman(batch, aggregate_features[name]) for batch in values]
        for name, values in batch_features.items()
    }
    metadata = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "manifest_sha256": args.manifest_sha256,
        "source_state_sha256": os.environ.get("SOURCE_STATE_SHA256"),
        "run_id": os.environ.get("EXPERIMENT_RUN_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "dataset": args.dataset,
        "split": args.split,
        "region": [args.region_start_fraction, args.region_end_fraction],
        "fit_offsets": fit_starts,
        "feature_offsets": feature_starts,
        "label_offsets": label_starts,
        "fit_batches": len(fit_batches),
        "feature_batches": len(feature_batches),
        "label_batches": len(label_batches),
        "batch_size": args.batch_size,
        "seq_len": config.block_size,
        "position_encoding": config.position_encoding,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "total_heads": total_heads,
        "decomp_fit_backend": args.decomp_fit_backend,
        "decomp_fit_steps": args.decomp_fit_steps,
        "decomp_max_kl": args.decomp_max_kl,
        "capture_dtype": controller.last_capture_dtype,
        "selector_phase_seconds": controller.last_selector_phases,
        "elapsed_seconds": time.time() - started,
        "primary_label": "dense_mean_delta_nll",
        "label_contract": (
            "paired full-model NLL on disjoint controller-validation windows; "
            "one exact post-softmax-mean head replacement at a time"
        ),
    }
    result = {"metadata": metadata, "ranking": ranking, "heads": rows}
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "head_metrics.csv", rows)
    np.savez_compressed(
        args.out_dir / "supporting_arrays.npz",
        contribution_similarity=features["similarity"].astype(np.float32),
        baseline_label_losses=np.asarray(baseline_losses, dtype=np.float32),
        dense_label_batch_deltas=dense_batch_deltas.astype(np.float32),
        compact_label_batch_deltas=np.asarray([
            compact_labels[(layer, head)]["batch_deltas"]
            if (layer, head) in compact_labels
            else [np.nan] * len(label_batches)
            for layer, head in pairs
        ], dtype=np.float32),
        batch_redundancy_max=features["batch_redundancy_max"].astype(np.float32),
        batch_contribution_rms=features["batch_contribution_rms"].astype(np.float32),
        batch_block_influence_cos=features[
            "batch_block_influence_cos"
        ].astype(np.float32),
        batch_block_influence_l2=features[
            "batch_block_influence_l2"
        ].astype(np.float32),
        fit_offsets=np.asarray(fit_starts, dtype=np.int64),
        feature_offsets=np.asarray(feature_starts, dtype=np.int64),
        label_offsets=np.asarray(label_starts, dtype=np.int64),
    )
    print(json.dumps(ranking, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

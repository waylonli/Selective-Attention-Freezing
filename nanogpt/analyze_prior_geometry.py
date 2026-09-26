#!/usr/bin/env python3
"""Compare saved fixed patterns with the saved post-softmax mean target.

For a random attention row A with mean mu and a deterministic fixed row p,
the excess squared risk over mu is ||p-mu||^2, while the excess expected
forward KL is KL(mu || p).  Both quantities can therefore be audited from the
actual stored patterns without recapturing dense attention matrices.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch


def sha256_file(path: Path, chunk_size: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_prefix(layer: int) -> str:
    return f"transformer.h.{layer}.main_block."


def load_checkpoint(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in payload:
        raise ValueError(f"checkpoint has no model state: {path}")
    return payload


def frozen_labels(state: dict[str, torch.Tensor], n_layer: int) -> list[str]:
    labels = []
    for layer in range(n_layer):
        mask = state[state_prefix(layer) + "dyn_frozen"].bool()
        labels.extend(
            f"L{layer}H{head}" for head in mask.nonzero(as_tuple=True)[0].tolist()
        )
    return labels


def normalized_pattern_rows(
    state: dict[str, torch.Tensor], layer: int, head: int, begin: int, end: int,
    *, seq_len: int, device: torch.device,
) -> torch.Tensor:
    prefix = state_prefix(layer)
    decomp_key = prefix + "dyn_decomp"
    is_decomposed = (
        decomp_key in state and bool(state[decomp_key][head].item())
    )
    if is_decomposed:
        alpha = state[prefix + "dyn_alpha"][head, :seq_len].to(
            device=device, dtype=torch.float32
        )
        rho = state[prefix + "dyn_rho"][head, :seq_len].to(
            device=device, dtype=torch.float32
        )
        row_index = torch.arange(begin, end, device=device)[:, None]
        key_index = torch.arange(seq_len, device=device)[None, :]
        distance = row_index - key_index
        valid = distance >= 0
        logits = alpha[None, :].expand(end - begin, -1).clone()
        logits[valid] += rho[distance[valid]]
        logits.masked_fill_(~valid, float("-inf"))
        rows = torch.softmax(logits, dim=-1)
    else:
        pattern_key = prefix + "dyn_pattern"
        if pattern_key not in state:
            raise KeyError(f"missing dense pattern {pattern_key}")
        rows = state[pattern_key][head, begin:end, :seq_len].to(
            device=device, dtype=torch.float32
        )
    rows = rows.clamp_min(0)
    return rows / rows.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def label_pair(label: str) -> tuple[int, int]:
    layer, head = label[1:].split("H")
    return int(layer), int(head)


def geometry_for_head(
    mean_state: dict[str, torch.Tensor], prior_state: dict[str, torch.Tensor],
    label: str, *, seq_len: int, chunk_rows: int, device: torch.device,
) -> dict[str, float]:
    layer, head = label_pair(label)
    l2_sum = kl_sum = hellinger_sum = tv_sum = 0.0
    max_row_sum_error = 0.0
    for begin in range(0, seq_len, chunk_rows):
        end = min(begin + chunk_rows, seq_len)
        mu = normalized_pattern_rows(
            mean_state, layer, head, begin, end,
            seq_len=seq_len, device=device,
        )
        p = normalized_pattern_rows(
            prior_state, layer, head, begin, end,
            seq_len=seq_len, device=device,
        )
        difference = p - mu
        l2_sum += float(difference.square().sum().item())
        positive = mu > 0
        kl = torch.where(
            positive,
            mu * (mu.clamp_min(1e-30).log() - p.clamp_min(1e-30).log()),
            torch.zeros_like(mu),
        )
        kl_sum += float(kl.sum().item())
        hellinger_sum += float(
            (0.5 * (mu.sqrt() - p.sqrt()).square().sum()).item()
        )
        tv_sum += float((0.5 * difference.abs().sum()).item())
        max_row_sum_error = max(
            max_row_sum_error,
            float((p.sum(-1) - 1.0).abs().max().item()),
        )
        del mu, p, difference, kl
    return {
        "excess_l2_per_matrix_entry": l2_sum / (seq_len * seq_len),
        "excess_forward_kl_per_row": kl_sum / seq_len,
        "hellinger_squared_per_row": hellinger_sum / seq_len,
        "total_variation_per_row": tv_sum / seq_len,
        "max_normalized_row_sum_error": max_row_sum_error,
    }


def read_head_variance(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        return {
            row["label"]: float(row["attention_variance"])
            for row in csv.DictReader(handle)
        }


def rows_for_ppl(path: Path) -> dict[tuple[str, float], dict[str, object]]:
    payload = json.loads(path.read_text())
    return {
        (str(row["prior"]), float(row["rate"])): row
        for row in payload["rows"]
    }


def find_head_order(manifest_paths: list[Path], expected_count: int) -> list[str]:
    candidates: list[list[str]] = []
    for path in manifest_paths:
        payload = json.loads(path.read_text())
        for arm in payload["arms"]:
            labels = arm.get("expected", {}).get("frozen_head_labels", [])
            if len(labels) == expected_count:
                candidates.append([str(label) for label in labels])
    if not candidates:
        raise RuntimeError("could not find the 75% fixed head order")
    if any(order != candidates[0] for order in candidates[1:]):
        raise RuntimeError("75% experiment manifests disagree on head order")
    return candidates[0]


def mean_dicts(rows: list[dict[str, object]], keys: Iterator[str]) -> dict[str, float]:
    keys = list(keys)
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    config = manifest["prior_geometry"]
    seq_len = int(config["seq_len"])
    n_layer = int(config["n_layer"])
    rates = [float(rate) for rate in config["rates"]]
    max_heads = round(max(rates) * int(config["total_heads"]))
    head_order = find_head_order(
        [Path(path) for path in config["experiment_manifests"]], max_heads
    )
    variance = read_head_variance(Path(config["head_metrics_csv"]))
    ppl = rows_for_ppl(Path(config["ppl_results_json"]))
    device = torch.device(args.device)

    sources = config["pattern_sources"]
    for prior, source in sources.items():
        path = Path(source["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if not args.skip_hash_check:
            actual = sha256_file(path)
            if actual != source["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch for {prior}")

    mean_source = sources["mean"]
    mean_payload = load_checkpoint(Path(mean_source["checkpoint"]))
    mean_state = mean_payload["model"]
    if set(frozen_labels(mean_state, n_layer)) != set(head_order):
        raise RuntimeError("mean checkpoint frozen set does not match manifest")

    metric_keys = (
        "excess_l2_per_matrix_entry",
        "excess_forward_kl_per_row",
        "hellinger_squared_per_row",
        "total_variation_per_row",
    )
    head_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for prior, source in sources.items():
        print(f"[geometry] loading {prior}", flush=True)
        if prior == "mean":
            prior_payload = mean_payload
            prior_state = mean_state
        else:
            prior_payload = load_checkpoint(Path(source["checkpoint"]))
            prior_state = prior_payload["model"]
        if set(frozen_labels(prior_state, n_layer)) != set(head_order):
            raise RuntimeError(f"{prior} frozen set does not match manifest")
        prior_head_rows = []
        for index, label in enumerate(head_order):
            metrics = geometry_for_head(
                mean_state, prior_state, label,
                seq_len=seq_len, chunk_rows=args.chunk_rows, device=device,
            )
            row = {
                "prior": prior,
                "rank_index": index,
                "head": label,
                "attention_variance": variance[label],
                **metrics,
            }
            prior_head_rows.append(row)
            head_rows.append(row)
            if (index + 1) % 12 == 0:
                print(f"[geometry] {prior}: {index + 1}/{len(head_order)} heads", flush=True)
        for rate in rates:
            count = round(rate * int(config["total_heads"]))
            selected = prior_head_rows[:count]
            aggregate = mean_dicts(selected, iter(metric_keys))
            aggregate["mean_attention_variance"] = sum(
                float(row["attention_variance"]) for row in selected
            ) / len(selected)
            aggregate["total_l2_risk_proxy"] = (
                aggregate["mean_attention_variance"]
                + aggregate["excess_l2_per_matrix_entry"]
            )
            ppl_row = ppl[(prior, rate)]
            aggregate_rows.append({
                "prior": prior,
                "rate": rate,
                "heads": count,
                "delta_ppl_pct": float(ppl_row["delta_ppl_pct"]),
                **aggregate,
            })
        if prior != "mean":
            del prior_state, prior_payload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "seq_len": seq_len,
        "rates": rates,
        "head_order": head_order,
        "metric_contract": {
            "excess_l2_per_matrix_entry": "||P-mu||_F^2 / T^2",
            "excess_forward_kl_per_row": "KL(mu || P), averaged over query rows",
            "hellinger_squared_per_row": "H^2(mu,P), averaged over query rows",
            "total_l2_risk_proxy": (
                "independent held-out attention-variance estimate plus stored-pattern "
                "excess L2 risk"
            ),
        },
        "aggregate": aggregate_rows,
        "heads": head_rows,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(json.dumps(aggregate_rows, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Compact, position-aware features for attention-pattern diversity analysis."""

import math

import torch


def stratified_query_positions(seq_len, n_queries):
    """Choose deterministic query positions with both early and uniform coverage."""
    if seq_len <= 0 or n_queries <= 0:
        raise ValueError("seq_len and n_queries must be positive")
    if n_queries >= seq_len:
        return torch.arange(seq_len, dtype=torch.long)
    anchors = [0]
    p = 1
    while p < seq_len and len(anchors) < max(2, n_queries // 3):
        anchors.append(p)
        p *= 2
    remaining = max(n_queries - len(anchors), 1)
    uniform = torch.linspace(0, seq_len - 1, remaining + 2).round().long().tolist()
    positions = sorted(set(anchors + uniform + [seq_len - 1]))
    if len(positions) > n_queries:
        # Keep early logarithmic anchors and evenly subsample the remainder.
        keep_early = positions[:min(len(anchors), n_queries // 2)]
        tail = positions[len(keep_early):]
        need = n_queries - len(keep_early)
        idx = torch.linspace(0, len(tail) - 1, need).round().long().tolist()
        positions = sorted(set(keep_early + [tail[i] for i in idx]))
    return torch.tensor(positions[:n_queries], dtype=torch.long)


def feature_names(n_query_bins=8, n_distance_bins=12, local_windows=(4, 16, 64)):
    names = []
    for qb in range(n_query_bins):
        prefix = f"qbin{qb}"
        names.extend(f"{prefix}/distance_bin_{i}" for i in range(n_distance_bins))
        names.extend([
            f"{prefix}/window_start_mass",
            f"{prefix}/normalized_entropy",
            f"{prefix}/top1_mass",
            f"{prefix}/expected_log_distance",
        ])
        names.extend(f"{prefix}/local_mass_{w}" for w in local_windows)
    return names


def per_query_feature_names(n_distance_bins=12, local_windows=(4, 16, 64)):
    names = [f"distance_bin_{i}" for i in range(n_distance_bins)]
    names.extend([
        "window_start_mass",
        "normalized_entropy",
        "top1_mass",
        "expected_log_distance",
    ])
    names.extend(f"local_mass_{w}" for w in local_windows)
    return names


def _aggregate_query_features(per_query, positions, seq_len, n_query_bins):
    """Average ``(B,H,Q,Fq)`` observations within coarse query bins."""
    sums = torch.zeros(
        *per_query.shape[:2], n_query_bins, per_query.shape[-1],
        device=per_query.device, dtype=per_query.dtype,
    )
    counts = torch.zeros(n_query_bins, device=per_query.device, dtype=per_query.dtype)
    query_bin = torch.clamp((positions * n_query_bins) // max(seq_len, 1),
                            max=n_query_bins - 1)
    for qb in range(n_query_bins):
        take = query_bin == qb
        if bool(take.any()):
            sums[:, :, qb] = per_query[:, :, take].mean(dim=2)
            counts[qb] = int(take.sum())
    if bool((counts == 0).any()):
        raise ValueError("query positions leave at least one query bin empty")
    return sums.flatten(start_dim=2)


def _per_query_features(att, query_positions, seq_len, n_distance_bins, local_windows):
    """Convert sampled-query attention to (B,H,Q,F_per_query) features."""
    if att.ndim != 4:
        raise ValueError("att must have shape (B,H,Q,T)")
    B, H, Q, T = att.shape
    if T != seq_len or query_positions.numel() != Q:
        raise ValueError("query positions/sequence length do not match attention")
    att = att.float()
    key = torch.arange(T, device=att.device)
    distance = query_positions.to(att.device)[:, None] - key[None, :]
    causal = distance >= 0
    distance = distance.clamp_min(0)

    # 0=self, 1=previous, 2=distance 2-3, 3=4-7, ...; final bin is overflow.
    buckets = torch.zeros_like(distance)
    positive = distance > 0
    buckets[positive] = torch.floor(torch.log2(distance[positive].float())).long() + 1
    buckets.clamp_(max=n_distance_bins - 1)
    parts = []
    for b in range(n_distance_bins):
        mask = (buckets == b) & causal
        parts.append((att * mask).sum(dim=-1))

    parts.append(att[..., 0])
    entropy = -(att.clamp_min(1e-12) * att.clamp_min(1e-12).log()).sum(dim=-1)
    raw_denom = torch.log(query_positions.to(att.device).float() + 1.0)
    entropy_denom = torch.where(raw_denom > 0, raw_denom, torch.ones_like(raw_denom))
    parts.append(entropy / entropy_denom)
    parts.append(att.max(dim=-1).values)
    log_distance = torch.log1p(distance.float()) / max(math.log1p(max(T - 1, 1)), 1.0)
    parts.append((att * log_distance).sum(dim=-1))
    for window in local_windows:
        parts.append((att * ((distance < window) & causal)).sum(dim=-1))
    return torch.stack(parts, dim=-1)


@torch.no_grad()
def qk_attention_features(q, k, *, query_positions, n_query_bins=8,
                          n_distance_bins=12, local_windows=(4, 16, 64),
                          query_chunk=16, return_per_query=False):
    """Extract compact features without materialising full ``T x T`` attention.

    Args:
        q, k: ``(B,H,T,D)`` scaled-dot-product query/key tensors.
        query_positions: positions to inspect, normally from
            :func:`stratified_query_positions`.
    Returns:
        ``(B,H,F)`` features and the ordered feature-name list.
    """
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have matching (B,H,T,D) shapes")
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    B, H, T, D = q.shape
    positions = query_positions.to(q.device).long()
    if bool((positions < 0).any()) or bool((positions >= T).any()):
        raise ValueError("query position out of range")
    per_query_dim = n_distance_bins + 4 + len(local_windows)
    query_features = torch.empty(
        B, H, positions.numel(), per_query_dim,
        device=q.device, dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(D)

    for start in range(0, positions.numel(), query_chunk):
        pos = positions[start:start + query_chunk]
        scores = (q[:, :, pos] @ k.transpose(-2, -1)) * scale
        causal = torch.arange(T, device=q.device)[None, :] <= pos[:, None]
        att = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        per_query = _per_query_features(att, pos, T, n_distance_bins, local_windows)
        query_features[:, :, start:start + len(pos)] = per_query

    aggregated = _aggregate_query_features(query_features, positions, T, n_query_bins)
    result = (aggregated, feature_names(
        n_query_bins=n_query_bins,
        n_distance_bins=n_distance_bins,
        local_windows=local_windows,
    ))
    if return_per_query:
        return result + (
            query_features,
            per_query_feature_names(n_distance_bins, local_windows),
        )
    return result

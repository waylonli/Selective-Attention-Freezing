"""
Dynamic per-head attention freezing — "gradual attention training" (Version A).

Periodically during training, measure each head's attention variance and freeze
the most STABLE (low-variance) unfrozen heads to a fixed pattern (their own
learned average = a prior, or a random pattern). Frozen heads skip softmax(QK^T)
(no gradient to Q/K; V stays trainable), so the final model has cheap heads ->
faster inference.

Which / how many heads freeze is EMERGENT (variance-driven), not a preset rate:
- a stability threshold tau, auto-calibrated from the variance distribution at the
  first check (percentiles p_lo..p_hi) and relaxed over training, decides freezing;
- a max_rate safety cap bounds the total fraction frozen;
- freezing is monotonic (a frozen head never thaws) in this version.

Requires the attention modules to be built with rand_attn_dynamic=True (see
CausalSelfAttention in model.py: set_capture / freeze_heads / dyn_frozen).
"""

import math
import os
import sys
import time

import torch

# the canonical decomposed-logits definition lives next to the kernel
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from kernel.fused_attn import decomp_logits  # noqa: E402


def relative_attention_prior(mean, eps=1e-12):
    """Fit a causal relative-distance kernel to an attention mean.

    The model family is ``P[i,j] = g[i-j] / sum_{d=0}^i g[d]``.  Within a
    row, normalization cancels in the log odds
    ``log P[i,i-d] - log P[i,i] = log g[d] - log g[0]``.  Averaging those
    log odds over every row containing distance ``d`` is therefore the
    least-squares estimator in log-odds space and exactly recovers a true
    relative kernel (up to scale).  This reduces ``T x T`` values to ``T``.
    """
    if mean.ndim != 2 or mean.size(0) != mean.size(1):
        raise ValueError("attention mean must be a square matrix")
    T = mean.size(0)
    self_mass = torch.diagonal(mean).clamp_min(eps)
    log_kernel = torch.zeros(T, dtype=mean.dtype, device=mean.device)
    for distance in range(1, T):
        diagonal = torch.diagonal(mean, offset=-distance).clamp_min(eps)
        reference = self_mass[distance:]
        log_kernel[distance] = (diagonal.log() - reference.log()).mean()
    distance_mass = (log_kernel - log_kernel.max()).exp()
    row = torch.arange(T, device=mean.device)[:, None]
    col = torch.arange(T, device=mean.device)[None, :]
    distance = (row - col).clamp_min(0)
    causal = col <= row
    pattern = distance_mass[distance].masked_fill(~causal, 0)
    return pattern / pattern.sum(-1, keepdim=True).clamp_min(eps)


def _centered_batch_moments(values):
    """Return count, mean, and centered M2 without ``E[x^2]-E[x]^2``.

    The subtraction is performed around the batch mean while the samples are
    still on their original device; the compact statistics are then promoted
    to float64 on CPU for stable cross-batch Chan merging.
    """
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must have a non-empty sample dimension")
    batch_mean = values.mean(dim=0)
    batch_m2 = (values - batch_mean).square().sum(dim=0)
    return (
        int(values.shape[0]),
        # Move the compact FP32 statistics before promoting them. At long
        # context, promoting on CUDA first would create a large FP64 device
        # temporary and transfer twice as many bytes to host memory.
        batch_mean.detach().cpu().double(),
        batch_m2.detach().cpu().double(),
    )


def _merge_moments(count, mean, m2, batch_count, batch_mean, batch_m2):
    """Merge two sets of centered moments using Chan's parallel formula."""
    if batch_count <= 0:
        return count, mean, m2
    if count == 0:
        return batch_count, batch_mean.clone(), batch_m2.clone()
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
    return total, merged_mean, merged_m2


def _centered_batch_reduced_moments(values, *, cpu_dtype=torch.float64):
    """Return a full mean but only per-head centered M2.

    Head selection consumes the variance averaged over attention-matrix cells,
    not the individual cell variances.  Keeping the centered sum of squares
    reduced over the last two dimensions is therefore exact while avoiding a
    second ``(n_head, T, T)`` CPU tensor.
    """
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("attention values must have shape (B, H, T, T)")
    if cpu_dtype not in (torch.float32, torch.float64):
        raise ValueError("cpu_dtype must be float32 or float64")
    if values.shape[0] == 1:
        # At 16K, one BF16 (H,T,T) capture is roughly 6 GiB. Computing even a
        # trivial size-one mean on CUDA would allocate another complete tensor
        # while all layers' captures are still resident. Transfer and cast in
        # one operation instead; the within-batch centered M2 is exactly zero.
        batch_mean = values[0].detach().to(device="cpu", dtype=cpu_dtype)
        batch_m2 = torch.zeros(values.shape[1], dtype=cpu_dtype)
    else:
        batch_mean_device = values.mean(dim=0)
        batch_m2_device = (
            values - batch_mean_device).square().sum(dim=(0, 2, 3))
        batch_mean = batch_mean_device.detach().to(
            device="cpu", dtype=cpu_dtype)
        batch_m2 = batch_m2_device.detach().to(
            device="cpu", dtype=cpu_dtype)
    return (
        int(values.shape[0]),
        batch_mean,
        batch_m2,
    )


def _merge_reduced_moments(count, mean, head_m2, batch_count,
                           batch_mean, batch_head_m2):
    """Chan-merge vector observations while retaining scalar M2 per head."""
    if batch_count <= 0:
        return count, mean, head_m2
    if count == 0:
        return batch_count, batch_mean.clone(), batch_head_m2.clone()
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    correction = delta.square().sum(dim=(-1, -2)) * (
        count * batch_count / total)
    merged_m2 = head_m2 + batch_head_m2 + correction
    return total, merged_mean, merged_m2


def _negative_entropy_sum(attention, *, eps=1e-12,
                          max_work_bytes=512 * 1024 ** 2,
                          work_device=None):
    """Sum ``p log p`` over samples, queries, and keys for every head.

    ``attention`` has shape ``(B, H, T, T)``. Query-row chunking bounds the
    FP32 normalization and ``xlogy`` temporaries, which is essential for 16K
    layer-streamed capture. Rows are normalized in FP32 before the reduction
    so the result is a probability-space statistic even when capture uses
    BF16.
    """
    if attention.ndim != 4 or attention.shape[0] == 0:
        raise ValueError("attention must have shape (B, H, T, T)")
    B, H, T, K = attention.shape
    if T != K:
        raise ValueError("attention matrices must be square")
    # The main work tensors are the FP32 probabilities and xlogy output.
    bytes_per_query = max(B * H * K * 4 * 2, 1)
    query_chunk = max(1, min(T, max_work_bytes // bytes_per_query))
    work_device = attention.device if work_device is None else work_device
    total = torch.zeros(H, dtype=torch.float64)
    for start in range(0, T, query_chunk):
        probs = attention[:, :, start:start + query_chunk, :].to(
            device=work_device, dtype=torch.float32)
        probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(eps))
        contribution = torch.xlogy(probs, probs).sum(dim=(0, 2, 3))
        total.add_(contribution.detach().double().cpu())
    return total


def _mean_forward_kl_from_entropy(sample_negative_entropy_sum, mean_attention,
                                  n_samples, *, work_device=None):
    """Return ``E_s mean_i KL(P_s[i] || Pbar[i])`` for every head.

    The identity ``E KL(P_s || Pbar) = E[sum P_s log P_s] -
    sum Pbar log Pbar`` avoids storing a second ``H x T x T`` statistic.
    """
    if mean_attention.ndim != 3:
        raise ValueError("mean_attention must have shape (H, T, T)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    T = mean_attention.shape[-1]
    sample_term = sample_negative_entropy_sum / (n_samples * T)
    mean_term = _negative_entropy_sum(
        mean_attention.unsqueeze(0), work_device=work_device) / T
    # Tiny negative values can arise from finite-precision accumulation even
    # though the population quantity is nonnegative.
    return (sample_term - mean_term).clamp_min(0).float()


def fit_dirichlet_concentration(mean, var, causal, *, scale=1.0,
                                min_concentration=1e-2, max_concentration=1e6,
                                eps=1e-8):
    """Fit one Dirichlet concentration per causal query row by moments.

    For a Dirichlet row with mean ``mu`` and total concentration ``c``,
    Summing ``Var[A_j] = mu_j (1 - mu_j) / (c + 1)`` over keys gives the
    pooled estimator ``c = (1 - ||mu||^2) / sum(var) - 1``. The returned
    tensor has shape ``(T,)``. Deterministic rows receive max_concentration.
    """
    if mean.shape != var.shape or mean.ndim != 2:
        raise ValueError("mean and var must be matching (T, T) tensors")
    if causal.shape != mean.shape:
        raise ValueError("causal mask must match mean")
    if scale <= 0 or min_concentration <= 0 or max_concentration < min_concentration:
        raise ValueError("invalid Dirichlet concentration bounds/scale")

    T = mean.shape[0]
    out = torch.empty(T, dtype=torch.float32)
    mask = causal.bool()
    for i in range(T):
        valid_keys = mask[i]
        mu = mean[i, valid_keys].double().clamp_min(0.0)
        mu = (mu + 1e-12) / (mu.sum() + 1e-12 * mu.numel())
        vv = var[i, valid_keys].double().clamp_min(0.0)
        numerator = 1.0 - mu.square().sum()
        total_var = vv.sum()
        if float(total_var) <= eps or float(numerator) <= eps:
            concentration = torch.tensor(max_concentration, dtype=torch.float64)
        else:
            concentration = numerator / total_var - 1.0
            if not bool(torch.isfinite(concentration)) or float(concentration) <= 0:
                concentration = torch.tensor(min_concentration, dtype=torch.float64)
        out[i] = (concentration * scale).clamp(min_concentration, max_concentration)
    return out


def sample_dirichlet_pattern(mean, var, causal, *, rng, scale=1.0,
                             min_concentration=1e-2, max_concentration=1e6,
                             alpha_min=1e-6):
    """Sample a causal row-stochastic pattern from fitted row-wise Dirichlets."""
    concentration = fit_dirichlet_concentration(
        mean, var, causal, scale=scale,
        min_concentration=min_concentration,
        max_concentration=max_concentration,
    )
    T = mean.shape[0]
    pattern = torch.zeros_like(mean, dtype=torch.float64)
    mask = causal.bool()
    alpha = torch.ones_like(pattern)
    deterministic = torch.zeros(T, dtype=torch.bool)
    for i in range(T):
        valid_keys = mask[i]
        mu = mean[i, valid_keys].double().clamp_min(0.0)
        mu = (mu + 1e-12) / (mu.sum() + 1e-12 * mu.numel())
        vv = var[i, valid_keys].double().clamp_min(0.0)
        deterministic[i] = mu.numel() == 1 or float(vv.sum()) <= 1e-8 \
            or float(1.0 - mu.square().sum()) <= 1e-8
        alpha[i, valid_keys] = (mu * concentration[i].double()).clamp_min(alpha_min)

    # torch.distributions does not accept a Generator. Draw one seed from the
    # controller-owned generator, then isolate the global RNG while sampling.
    draw_seed = int(torch.randint(0, 2**31 - 1, (), generator=rng).item())
    with torch.random.fork_rng(devices=[]):
        # The Gamma tensors live on CPU. ``torch.manual_seed`` would also
        # reseed every accelerator and leak into dropout/sampling during
        # training; seed only the forked CPU default generator instead.
        torch.default_generator.manual_seed(draw_seed)
        gamma = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
    gamma.masked_fill_(~mask, 0.0)
    pattern = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(1e-300)
    pattern[deterministic] = mean.double()[deterministic]
    pattern.masked_fill_(~mask, 0.0)
    pattern = pattern / pattern.sum(dim=-1, keepdim=True).clamp_min(1e-300)
    return pattern.float(), concentration, deterministic


def decomp_stats(P, causal):
    """Sufficient statistics of the (row-normalized) mean pattern for the
    decomposed alpha+rho fit:  c[j] = sum_i Pn[i, j]  (absolute-position /
    column mass, (N, T)) and  d[k] = sum_{i-j=k} Pn[i, j]  (relative-distance /
    diagonal mass, (N, T)). These two T-vectors are ALL the fit ever needs —
    the T x T matrix can be discarded (or, later, never materialized)."""
    N, T = P.shape[0], P.shape[-1]
    m = causal.bool()
    Pn = P.clamp_min(1e-9) * causal
    Pn = Pn / Pn.sum(-1, keepdim=True)
    c = (Pn * causal).sum(-2)                                     # (N, T)
    idx = torch.arange(T, device=P.device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)
    didx = dist.view(-1)[m.view(-1)]                              # (nnz,)
    vals = Pn.reshape(N, -1)[:, m.view(-1)]                       # (N, nnz)
    d = torch.zeros(N, T, device=P.device).index_add_(1, didx, vals)
    return c, d


def fit_alpha_rho_from_stats(c, d, causal, steps=400, lr=0.05):
    """Batched convex MLE of P_hat = row-softmax(alpha[j] + rho[i-j]) from the
    sufficient statistics ONLY (never touches the T x T mean):

        CE(alpha, rho) = ( sum_i logZ_i - <c, alpha> - <d, rho> ) / T

    with logZ_i = logsumexp_{j<=i}(alpha[j] + rho[i-j]). The gradient is
    IDENTICAL to the full-matrix cross-entropy fit (c, d are its only
    couplings to P), so this converges to the same Q. c, d: (N, T).
    Returns (alpha, rho, z) with z the converged normalizers (natural log) —
    callers can compute the exact fit CE/KL from the stats alone with it.
    The in-loop logit expression follows kernel.fused_attn.decomp_logits (kept
    inline so `dist` is built once, not per Adam step)."""
    N, T = c.shape
    device = c.device
    idx = torch.arange(T, device=device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)
    m = causal.bool()
    # init rho from the mean probability per diagonal (log of d / diag length)
    dcnt = (T - idx).clamp_min(1).float()
    alpha = torch.zeros(N, T, device=device, requires_grad=True)
    rho = (d / dcnt).clamp_min(1e-9).log().detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([alpha, rho], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            L = alpha[:, None, :] + rho[:, dist]
            L = L.masked_fill(~m, float("-inf"))
            z = torch.logsumexp(L, dim=-1)                        # (N, T)
            loss = ((z.sum(-1) - (c * alpha).sum(-1) - (d * rho).sum(-1)) / T).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    alpha, rho = alpha.detach(), rho.detach()
    with torch.no_grad():
        z = torch.logsumexp(decomp_logits(alpha, rho), dim=-1)
    return alpha, rho, z


def decomp_log_normalizers_fft(alpha, rho):
    """Compute causal alpha+rho row log-normalizers by linear convolution.

    ``Z[i] = sum_{j<=i} exp(alpha[j]) exp(rho[i-j])`` is the first ``T``
    entries of a convolution. Detached global shifts keep the exponentials in
    range without changing the exact derivative with respect to alpha/rho.
    """
    if alpha.shape != rho.shape or alpha.ndim != 2:
        raise ValueError("alpha and rho must have matching (N, T) shapes")
    _, T = alpha.shape
    n_fft = 1 << (2 * T - 1).bit_length()
    # A single shift is taken over the complete sequence, while early causal
    # prefixes may contain values far below a late-position maximum. In
    # float32, FFT round-off from those large late terms can then dominate a
    # small early-prefix convolution. The fitter is a one-time offline step,
    # so accumulate the convolution in float64 and cast the normalizers back
    # to the parameter dtype. This preserves the O(T log T) algorithm while
    # making its local row probabilities agree with stable dense logsumexp.
    work_dtype = (torch.float64 if alpha.dtype in
                  (torch.float16, torch.bfloat16, torch.float32)
                  else alpha.dtype)
    alpha_work = alpha.to(work_dtype)
    rho_work = rho.to(work_dtype)
    alpha_shift = alpha_work.detach().amax(dim=-1, keepdim=True)
    rho_shift = rho_work.detach().amax(dim=-1, keepdim=True)
    a = (alpha_work - alpha_shift).exp()
    r = (rho_work - rho_shift).exp()
    spectrum = torch.fft.rfft(a, n=n_fft) * torch.fft.rfft(r, n=n_fft)
    convolution = torch.fft.irfft(spectrum, n=n_fft)[..., :T]
    tiny = torch.finfo(convolution.dtype).tiny
    z = convolution.clamp_min(tiny).log() + alpha_shift + rho_shift
    return z.to(alpha.dtype)


def fit_alpha_rho_from_stats_fft(c, d, steps=400, lr=0.05):
    """FFT candidate for the same convex MLE as the dense normalizer fit.

    It remains a separate function until paper-checkpoint comparisons establish
    numerical and fit-KL equivalence on both CPU and CUDA.
    """
    if c.shape != d.shape or c.ndim != 2:
        raise ValueError("c and d must have matching (N, T) shapes")
    N, T = c.shape
    device = c.device
    idx = torch.arange(T, device=device)
    dcnt = (T - idx).clamp_min(1).float()
    alpha = torch.zeros(N, T, device=device, requires_grad=True)
    rho = (d / dcnt).clamp_min(1e-9).log().detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([alpha, rho], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            z = decomp_log_normalizers_fft(alpha, rho)
            loss = ((z.sum(-1) - (c * alpha).sum(-1)
                     - (d * rho).sum(-1)) / T).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    alpha, rho = alpha.detach(), rho.detach()
    with torch.no_grad():
        z = decomp_log_normalizers_fft(alpha, rho)
    return alpha, rho, z


def decomp_materialize_q(alpha, rho, causal):
    """(N, T, T) row-softmax(alpha[j] + rho[i-j]) from fitted (N, T) vectors."""
    del causal  # the canonical logits are already causally masked
    return torch.softmax(decomp_logits(alpha, rho), dim=-1)


def kl_rows_batched(Ps, Qs, causal, eps=1e-9):
    """Per-head mean-over-rows KL(P_i || Q_i) for stacked (N, T, T) -> (N,)."""
    m = causal.bool()
    Pn = Ps.clamp_min(eps) * causal
    Pn = Pn / Pn.sum(-1, keepdim=True)
    Qn = Qs.clamp_min(eps) * causal
    Qn = Qn / Qn.sum(-1, keepdim=True)
    con = torch.where(m, Pn.clamp_min(eps) * (Pn.clamp_min(eps).log()
                                              - Qn.clamp_min(eps).log()),
                      torch.zeros_like(Pn))
    return con.sum(-1).mean(-1)


def _pct(sorted_vals, p):
    """Linear-interpolated p-th percentile of an already-sorted list."""
    if not sorted_vals:
        return float("inf")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k); c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def parse_fixed_head_order(raw, n_layer, n_head):
    """Parse a comma-separated ``L<li>H<h>`` intervention order.

    The explicit order is a paper-experiment control: matched prior-content
    arms must replace exactly the same routing slots, rather than independently
    rerunning a selector and merely hoping that its result is unchanged.
    """
    if not raw or not raw.strip():
        return []
    flat = []
    for label in raw.split(","):
        label = label.strip()
        if not label.startswith("L") or "H" not in label:
            raise ValueError(
                f"invalid fixed head label {label!r}; expected L<li>H<h>")
        layer_raw, head_raw = label[1:].split("H", 1)
        try:
            layer, head = int(layer_raw), int(head_raw)
        except ValueError as exc:
            raise ValueError(
                f"invalid fixed head label {label!r}; expected L<li>H<h>") from exc
        if not 0 <= layer < n_layer or not 0 <= head < n_head:
            raise ValueError(
                f"fixed head {label!r} is outside {n_layer}x{n_head} model")
        flat.append(layer * n_head + head)
    if len(flat) != len(set(flat)):
        raise ValueError("freeze_fixed_head_order contains duplicate heads")
    return flat


def get_attn_modules(model):
    """Return the CausalSelfAttention modules (in layer order) that support
    dynamic freezing."""
    mods = []
    for block in model.transformer.h:
        m = block.main_block
        if hasattr(m, "block"):   # StatefulBlock wrapper
            m = m.block
        if getattr(m, "dynamic", False):
            mods.append(m)
    return mods


class DynamicFreezeController:
    def __init__(self, model, *, content="own_prior", mode="global", check_interval=500,
                 warmup_frac=0.1, max_rate=1.0, ramp_end=1.0, p_lo=10.0, p_hi=60.0,
                 measure_batches=16, val_tol=0.0, val_rel=0.0, val_batches=8,
                 select_signal="variance", grad_beta=0.9, sig_t=2.0, plateau_z=1.5,
                 variance_estimator="sample", dirichlet_scale=1.0,
                 dirichlet_min_concentration=1e-2, dirichlet_max_concentration=1e6,
                 decomp_max_kl=0.2, decomp_fit_steps=400,
                 decomp_fit_backend="dense", random_logit_std=1.0,
                 fixed_head_order="", allow_dense_fallback=False,
                 seed=42, log=None, measurement_forward=None):
        self.model = model
        self.mods = get_attn_modules(model)
        self.content = content              # 'own_prior' | 'own_prior_decomposed' | 'random' | ...
        self.mode = mode                    # 'global'|'per_layer'|'threshold'|'select_once'|
                                            # 'significance'|'relative'|'ramp'
        self.M = check_interval
        # 'ramp' mode reaches max_rate at ramp_end (fraction of training) rather than at
        # the very end, so late-frozen heads still get [ramp_end,1] of training to re-adapt
        # (freezing too late leaves the value path no time to recover -> higher loss).
        self.ramp_end = ramp_end
        self.warmup_frac = warmup_frac
        self.max_rate = max_rate            # safety cap on fraction frozen (1.0 = off)
        self.p_lo, self.p_hi = p_lo, p_hi
        self.K = measure_batches
        if variance_estimator not in ("sample", "batch_mean"):
            raise ValueError("variance_estimator must be 'sample' or 'batch_mean'")
        self.variance_estimator = variance_estimator
        self.dirichlet_scale = dirichlet_scale
        self.dirichlet_min_concentration = dirichlet_min_concentration
        self.dirichlet_max_concentration = dirichlet_max_concentration
        # val-loss guard (Marcio): only commit a freeze while the TOTAL val-loss
        # increase it causes stays under val_tol (cumulative budget, in nats of NLL).
        # val_tol = 0 disables the guard (rate then governed by variance + max_rate).
        self.val_tol = val_tol
        # RELATIVE budget (mode='relative'): cumulative val-PPL increase from freezing may
        # reach at most val_rel (unitless, e.g. 0.01 = 1% PPL). Portable across model sizes
        # (no absolute nats); permits the small unavoidable cost 'significance' forbade.
        self.val_rel = val_rel
        self.val_batches = val_batches
        self.cum_damage = 0.0
        self.last_var = None     # most recent per-head variance (n_layer, n_head)
        self.last_kl = None      # E_s mean_i KL(P_s[i] || Pbar[i]) in nats
        self.last_capture_dtype = None
        self._lm = self._ls = None   # per-position logit mean/std (for the gaussian prior)
        self._attn_var = None        # true per-sample attention variance (for Dirichlet)
        self.last_dirichlet_concentration = None
        self.last_dirichlet_fallback = None
        # content='own_prior_decomposed': frozen priors are stored as alpha+rho
        # logit VECTORS (O(T)/head, fitted from the (c, d) sufficient statistics).
        # Heads whose fit KL exceeds decomp_max_kl are SKIPPED for the round
        # (not dense-frozen — that would silently reintroduce T^2 storage).
        repr_ = getattr(self.mods[0], "prior_repr", "dense") if self.mods else "dense"
        decomp_contents = {
            "own_prior_decomposed", "random_decomposed", "uniform_decomposed"
        }
        if content in decomp_contents and repr_ != "decomposed":
            raise ValueError(f"content={content!r} requires the model to be built with "
                             "rand_attn_prior_repr='decomposed'")
        if (repr_ == "decomposed" and content not in decomp_contents
                and not allow_dense_fallback):
            raise ValueError(
                f"rand_attn_prior_repr='decomposed' with freeze_content='{content}' "
                "would allocate a dense T x T fallback; choose a decomposed prior "
                "or explicitly enable the diagnostic-only dense oracle")
        self.allow_dense_fallback = bool(allow_dense_fallback)
        self.decomp_max_kl = decomp_max_kl
        self.decomp_fit_steps = decomp_fit_steps
        if decomp_fit_backend not in ("dense", "fft"):
            raise ValueError("decomp_fit_backend must be 'dense' or 'fft'")
        self.decomp_fit_backend = decomp_fit_backend
        self.random_logit_std = random_logit_std
        # A full, seed-fixed bank makes a head's structured-random prior
        # independent of the requested rate, pair order, and freeze schedule.
        # This is required for controlled rate/time/schedule comparisons.
        self._random_decomp_banks = {}
        self.decomp_skipped = 0      # cumulative count of bad-fit skips (logged)
        self.last_decomp_kl = {}     # (li, h) -> fit KL of the most recent fit
        # Populated for one-shot selection without changing its decisions. The
        # compact c/d statistics make alternative fitters reproducible offline.
        self.last_selector_phases = {}
        self.last_decomp_stats_chunks = []
        self._profile_selector_phases = False
        self.sig_t = sig_t           # t-threshold for significance-gated freezing (unitless)
        valid_select_signals = {
            "variance", "kl", "gradient", "combined", "combined_veto"
        }
        if select_signal not in valid_select_signals:
            raise ValueError(
                f"unknown select_signal {select_signal!r}; expected one of "
                f"{sorted(valid_select_signals)}")
        self.select_signal = select_signal
        self.grad_beta = grad_beta
        self.grad_ema = None     # EMA of per-head Q/K grad-norm (convergence signal)
        self.rng = torch.Generator().manual_seed(seed)
        self.log = log or (lambda *a, **k: print(*a, flush=True, **k))
        # Optional task-specific capture forward. The callback receives exactly
        # what ``get_batch('train')`` returns and runs under ``ctx`` + no_grad.
        # Language-model training keeps the historical (X, Y) default below.
        self.measurement_forward = measurement_forward
        self.tau_lo = self.tau_hi = None
        self.n_layer = len(self.mods)
        self.n_head = int(self.mods[0].dyn_frozen.numel()) if self.mods else 0
        self.total_heads = self.n_layer * self.n_head
        # Stream capture when retaining every layer's FP32 attention would
        # exceed this budget. Basing the decision on L*H*T^2 covers both
        # 124M/16K and wider/deeper 1B/8K models.
        self.capture_stream_max_resident_bytes = 64 * 1024 ** 3
        self.fixed_head_order = parse_fixed_head_order(
            fixed_head_order, self.n_layer, self.n_head)
        if self.fixed_head_order and mode not in {"select_once", "ramp"}:
            raise ValueError(
                "freeze_fixed_head_order is supported only by select_once/ramp")

        # --- mode='plateau': train-loss-stabilization-gated auto controller state ---
        # Only dimensionless constants (same class as grad_beta/sig_t); everything with loss
        # units is self-calibrated online. See observe_loss().
        self.pl_z = plateau_z      # z-score for the plateau band, power floor, and harm test
                                   # (also the main timing knob: higher z -> earlier freezing)
        self.pl_rho = 4            # slow/fast EMA span ratio
        self.pl_inc_cap = 0.02     # max increment as a fraction of all heads
        self.pl_harm_max = 2       # stop after this many harmful (rolled-back) freezes
        self.pl_buf = []; self.pl_S = None
        self.pl_af = self.pl_as = self.pl_av = None
        self.pl_f = self.pl_s = None; self.pl_v = 0.0
        self.pl_drate = 0.0        # EMA of per-step fast-EMA descent (>0 while improving)
        self.pl_g_ref = 0.0; self.pl_g_ref_since = 0   # running max smoothed slope + staleness
        self.pl_started = False; self.pl_armed = False; self.pl_state = "BURN"
        self.pl_f_before = None; self.pl_drate_freeze = 0.0; self.pl_t_freeze = -1
        self.pl_last_inc = []; self.pl_last_bar_min = float("inf"); self.pl_bar = float("inf")
        self.pl_adapt_times = []; self.pl_harm_count = 0; self.pl_stopped = False

    @torch.no_grad()
    def _val_loss(self, get_batch, ctx, n=None):
        """Mean val loss over n batches (model in eval mode)."""
        n = n or self.val_batches
        was = self.model.training
        self.model.eval()
        tot = 0.0
        for _ in range(n):
            X, Y = get_batch("val")
            with ctx:
                _, loss = self.model(X, Y)
            tot += float(loss)
        self.model.train(was)
        return tot / n

    def current_rate(self):
        f = sum(int(m.dyn_frozen.sum()) for m in self.mods)
        return f / max(self.total_heads, 1)

    def _phase_clock(self):
        """Synchronised timestamp for one-time selector phase accounting."""
        if self.mods and self.mods[0].dyn_frozen.device.type == "cuda":
            torch.cuda.synchronize(self.mods[0].dyn_frozen.device)
        return time.perf_counter()

    def _phase_add(self, name, start):
        if not self._profile_selector_phases:
            return
        elapsed = self._phase_clock() - start
        self.last_selector_phases[name] = (
            self.last_selector_phases.get(name, 0.0) + elapsed)

    @torch.no_grad()
    def _measure(self, get_batch, ctx):
        """Run K batches with capture on; return per-head variance (n_layer,n_head)
        and each layer's mean attention (n_head,T,T).

        ``variance_estimator='sample'`` computes the actual variance across all
        K*B examples. ``batch_mean`` preserves the historical estimator (variance
        across K batch means) for exact reproduction of older experiments.
        """
        need_attn_cell_var = self.content == "own_prior_dirichlet"
        need_logits = self.content in ("own_prior_sharp", "own_prior_gaussian")
        need_logit_var = self.content == "own_prior_gaussian"
        need_legacy = self.variance_estimator == "batch_mean"

        configured_context = int(self.mods[0]._dyn_causal.size(-1))
        estimated_capture_bytes = (
            self.n_layer * self.n_head * configured_context ** 2 * 4)
        if estimated_capture_bytes > self.capture_stream_max_resident_bytes:
            if need_attn_cell_var or need_logits or need_legacy:
                raise RuntimeError(
                    "layer-streamed long-context capture currently supports "
                    "mean priors with variance_estimator='sample' only")
            self.log(
                "[freeze] layer-streamed capture: estimated full retention "
                f"{estimated_capture_bytes / 1024 ** 3:.1f} GiB")
            return self._measure_layer_streamed(
                get_batch, ctx, accumulator_dtype=torch.float32)

        was_training = self.model.training
        self.model.eval()
        for m in self.mods:
            m.set_capture(True, capture_logits=need_logits)
        att_mean = [None] * self.n_layer   # float64 CPU Chan/Welford statistics
        # Dirichlet needs variance for every cell. Other priors need only the
        # exact per-head mean variance, so retain a reduced (H,) M2.
        att_m2 = [None] * self.n_layer
        sample_negative_entropy = [
            torch.zeros(self.n_head, dtype=torch.float64)
            for _ in range(self.n_layer)
        ]
        logit_mean = [None] * self.n_layer if need_logits else None
        logit_m2 = [None] * self.n_layer if need_logit_var else None
        legacy_mean = [None] * self.n_layer if need_legacy else None
        legacy_m2 = [None] * self.n_layer if need_legacy else None
        n_samp = 0
        n_logit = 0
        n_batches = 0
        try:
            for _ in range(self.K):
                batch = get_batch("train")
                with ctx:
                    if self.measurement_forward is None:
                        X, Y = batch
                        self.model(X, Y)
                    else:
                        self.measurement_forward(batch)
                n_batches += 1
                batch_size = None
                for li, m in enumerate(self.mods):
                    if self.last_capture_dtype is None:
                        self.last_capture_dtype = str(m._last_attn.dtype)
                    att = m._last_attn                           # (B, n_head, T, T)
                    sample_negative_entropy[li].add_(
                        _negative_entropy_sum(att))
                    if need_attn_cell_var:
                        bn, bm, b2 = _centered_batch_moments(att)
                        count = n_samp if att_mean[li] is not None else 0
                        _, att_mean[li], att_m2[li] = _merge_moments(
                            count, att_mean[li], att_m2[li], bn, bm, b2)
                    else:
                        # Full FP64 means are useful at ordinary context sizes,
                        # but require about 309 GiB at 16K. FP32 accumulation
                        # retains the exact statistic definition in 154 GiB and
                        # avoids an otherwise unnecessary precision-driven OOM.
                        accumulator_dtype = (
                            torch.float32 if att.shape[-1] >= 16384
                            else torch.float64)
                        bn, bm, b2 = _centered_batch_reduced_moments(
                            att, cpu_dtype=accumulator_dtype)
                        count = n_samp if att_mean[li] is not None else 0
                        _, att_mean[li], att_m2[li] = _merge_reduced_moments(
                            count, att_mean[li], att_m2[li], bn, bm, b2)
                    # A complete 16K capture is about 6 GiB per layer in BF16.
                    # Release it as soon as its CPU moments have been merged so
                    # the allocator can reuse the storage on the next batch.
                    m._last_attn = None
                    batch_size = bn
                    if need_legacy:
                        # Historical estimator: variance across equal-weight
                        # batch means, reduced exactly to one M2 per head.
                        _, legacy_mean[li], legacy_m2[li] = _merge_reduced_moments(
                            n_batches - 1, legacy_mean[li], legacy_m2[li],
                            1, bm, torch.zeros(self.n_head, dtype=bm.dtype))
                    if need_logits:
                        lg = m._last_logits.float()              # raw pre-mask logits
                        if need_logit_var:
                            lbn, lbm, lb2 = _centered_batch_moments(lg)
                            lcount = n_logit if logit_mean[li] is not None else 0
                            _, logit_mean[li], logit_m2[li] = _merge_moments(
                                lcount, logit_mean[li], logit_m2[li], lbn, lbm, lb2)
                        else:
                            lbn = int(lg.shape[0])
                            lbm = lg.mean(dim=0).detach().double().cpu()
                            lcount = n_logit if logit_mean[li] is not None else 0
                            if lcount == 0:
                                logit_mean[li] = lbm.clone()
                            else:
                                total = lcount + lbn
                                logit_mean[li].add_(
                                    (lbm - logit_mean[li]) * (lbn / total))
                n_samp += batch_size
                if need_logits:
                    n_logit += lbn
        finally:
            for m in self.mods:
                m.set_capture(False)
            self.model.train(was_training)
        # Convert one layer at a time and release its FP64 accumulator before
        # moving on. At T=8192 the complete FP64 means are 72 GiB and the FP32
        # return values are 36 GiB; retaining both complete lists at once is an
        # avoidable host-memory peak.
        means = []
        for li in range(self.n_layer):
            means.append(att_mean[li].float())
            att_mean[li] = None
        if need_attn_cell_var:
            sample_var = [
                (att_m2[li] / max(n_samp - 1, 1)).clamp_min(0).float()
                for li in range(self.n_layer)
            ]
            sample_head_var = torch.stack([
                sample_var[li].mean(dim=(-1, -2)) for li in range(self.n_layer)
            ])
        else:
            sample_var = None
            cells = means[0].shape[-1] * means[0].shape[-2]
            sample_head_var = torch.stack([
                (att_m2[li] / max(n_samp - 1, 1) / cells).clamp_min(0).float()
                for li in range(self.n_layer)
            ])
        if self.variance_estimator == "sample":
            per_head_var = sample_head_var
        else:
            cells = means[0].shape[-1] * means[0].shape[-2]
            per_head_var = torch.stack([
                (legacy_m2[li] / max(n_batches - 1, 1) / cells).clamp_min(0).float()
                for li in range(self.n_layer)
            ])
        self.last_var = per_head_var
        self.last_kl = torch.stack([
            _mean_forward_kl_from_entropy(
                sample_negative_entropy[li], means[li], n_samp,
                work_device=self.mods[0].dyn_frozen.device)
            for li in range(self.n_layer)
        ])
        self._attn_var = sample_var
        # Per-position logit statistics exist only for priors that consume them.
        self._lm = ([logit_mean[li].float() for li in range(self.n_layer)]
                    if need_logits else None)
        self._ls = ([
            (logit_m2[li] / max(n_logit - 1, 1)).clamp_min(0).sqrt().float()
            for li in range(self.n_layer)
        ] if need_logit_var else None)
        return per_head_var, means

    @torch.no_grad()
    def _measure_layer_streamed(self, get_batch, ctx, *, accumulator_dtype):
        """Measure long contexts while retaining one layer at a time.

        The same K batches are replayed for every layer. All layers use the
        eager attention path on every replay, matching ordinary capture; only
        the target layer retains its dense matrix. This preserves the measured
        statistic and avoids the full-model ``L * H * T * T`` GPU residency.
        """
        batches = [get_batch("train") for _ in range(self.K)]
        was_training = self.model.training
        self.model.eval()
        att_mean = [None] * self.n_layer
        att_m2 = [None] * self.n_layer
        sample_negative_entropy = [
            torch.zeros(self.n_head, dtype=torch.float64)
            for _ in range(self.n_layer)
        ]
        counts = [0] * self.n_layer
        try:
            for li, target in enumerate(self.mods):
                layer_t0 = time.perf_counter()
                self.log(f"[freeze] capture layer {li + 1}/{self.n_layer}: "
                         f"{len(batches)} calibration batches")
                for module in self.mods:
                    module.set_capture(
                        True, capture_logits=False, retain_attention=False)
                target.set_capture(
                    True, capture_logits=False, retain_attention=True)
                count = 0
                for batch in batches:
                    with ctx:
                        if self.measurement_forward is None:
                            X, Y = batch
                            self.model(X, Y)
                        else:
                            self.measurement_forward(batch)
                    att = target._last_attn
                    if att is None:
                        raise RuntimeError(
                            f"layer {li} did not retain attention during capture")
                    if self.last_capture_dtype is None:
                        self.last_capture_dtype = str(att.dtype)
                    sample_negative_entropy[li].add_(
                        _negative_entropy_sum(att))
                    bn, bm, b2 = _centered_batch_reduced_moments(
                        att, cpu_dtype=accumulator_dtype)
                    _, att_mean[li], att_m2[li] = _merge_reduced_moments(
                        count, att_mean[li], att_m2[li], bn, bm, b2)
                    count += bn
                    target._last_attn = None
                counts[li] = count
                self.log(f"[freeze] capture layer {li + 1}/{self.n_layer} complete "
                         f"in {time.perf_counter() - layer_t0:.1f}s")
        finally:
            for module in self.mods:
                module.set_capture(False)
            self.model.train(was_training)

        if not counts or min(counts) <= 0 or len(set(counts)) != 1:
            raise RuntimeError(f"inconsistent streamed capture counts: {counts}")
        n_samp = counts[0]
        means = []
        for li in range(self.n_layer):
            means.append(att_mean[li].float())
            att_mean[li] = None
        cells = means[0].shape[-1] * means[0].shape[-2]
        per_head_var = torch.stack([
            (att_m2[li] / max(n_samp - 1, 1) / cells).clamp_min(0).float()
            for li in range(self.n_layer)
        ])
        self.last_var = per_head_var
        self.last_kl = torch.stack([
            _mean_forward_kl_from_entropy(
                sample_negative_entropy[li], means[li], n_samp,
                work_device=self.mods[0].dyn_frozen.device)
            for li in range(self.n_layer)
        ])
        self._attn_var = None
        self._lm = self._ls = None
        return per_head_var, means

    @torch.no_grad()
    def per_head_grad_norm(self):
        """Per-head Q/K gradient norm (n_layer, n_head), normalized by the mean across
        heads at this step to cancel the LR schedule. Returns None if grads aren't ready.
        A complementary signal to variance: low value = the head has ~stopped learning."""
        rows = []
        for m in self.mods:
            if m.q_proj.weight.grad is None:
                return None
            hd = m.head_dim
            qn = m.q_proj.weight.grad.view(m.n_head, hd, -1).norm(dim=(1, 2))
            kn = m.k_proj.weight.grad.view(m.n_head, hd, -1).norm(dim=(1, 2))
            rows.append((qn + kn).detach().float().cpu())
        g = torch.stack(rows)                       # (n_layer, n_head)
        mean = g.mean().clamp(min=1e-12)
        return g / mean                             # relative grad norm

    @torch.no_grad()
    def update_grad_ema(self):
        """Call every training step (after backward): maintain an EMA of the per-head
        Q/K grad-norm, the convergence signal used by 'gradient'/'combined' selection."""
        g = self.per_head_grad_norm()
        if g is None:
            return
        self.grad_ema = g if self.grad_ema is None else \
            self.grad_beta * self.grad_ema + (1 - self.grad_beta) * g

    @torch.no_grad()
    def _freeze_select_once(self, iter_num, get_batch, ctx):
        """Iso-rate ABLATION of the selection signal: at the first check past warmup,
        freeze exactly max_rate of heads chosen by select_signal, once, to their priors.
        variance and KL are probability-space input-dependence measures;
        gradient measures convergence; combined is low in BOTH (rank-sum)."""
        if self.current_rate() > 0:
            return
        self.last_selector_phases = {}
        self.last_decomp_stats_chunks = []
        self._profile_selector_phases = True
        capture_t0 = self._phase_clock()
        phv, means = self._measure(get_batch, ctx)
        self._phase_add("capture_and_mean_s", capture_t0)
        ranking_t0 = self._phase_clock()
        n_freeze = int(self.max_rate * self.total_heads)
        var = phv.flatten()
        if self.select_signal == "kl":
            if self.last_kl is None:
                raise RuntimeError("KL selection requested without measured KL scores")
            key, sig = self.last_kl.flatten(), "forward_kl"
        elif self.select_signal == "gradient" and self.grad_ema is not None:
            key, sig = self.grad_ema.flatten(), "gradient"
        elif self.select_signal == "combined" and self.grad_ema is not None:
            rv = var.argsort().argsort().float()               # variance rank (0 = lowest)
            rg = self.grad_ema.flatten().argsort().argsort().float()
            key, sig = rv + rg, "combined"                     # low in BOTH (symmetric rank-sum)
        elif self.select_signal == "combined_veto" and self.grad_ema is not None:
            # asymmetric: variance SELECTS (lowest first), gradient VETOES still-learning heads
            g = self.grad_ema.flatten()
            eligible = g <= g.median()                         # converged half (not vetoed)
            key = var.argsort().argsort().float()              # variance order
            key = key + (~eligible).float() * self.total_heads  # vetoed heads chosen only if needed
            sig = "combined_veto"
        else:
            key, sig = var, ("variance" if self.select_signal == "variance" else
                             self.select_signal + "(grad NA->variance)")
        order = key.argsort().tolist()                         # lowest-key first
        fixed = bool(self.fixed_head_order)
        if fixed:
            if len(self.fixed_head_order) < n_freeze:
                raise ValueError(
                    f"fixed head order has {len(self.fixed_head_order)} entries, "
                    f"but this arm requests {n_freeze}")
            order = self.fixed_head_order[:n_freeze]
            sig = "fixed_head_order"
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        self._phase_add("ranking_s", ranking_t0)
        # iso-rate contract: if decomposed fitting SKIPS bad-fit heads, top the
        # quota up from the next-lowest-signal candidates (this mode never runs
        # again, so skips would otherwise permanently undershoot the rate)
        froze, cursor = [], 0
        freeze_t0 = self._phase_clock()
        try:
            if fixed:
                pairs = [
                    (idx // self.n_head, idx % self.n_head) for idx in order
                ]
                froze = self._freeze_batch(pairs, means, T, causal)
                # The compact installer batches by layer and may therefore
                # return the exact requested slots in a different order. The
                # paper contract is set equality: any KL skip/substitution
                # changes the length or set and still fails closed.
                if len(froze) != len(pairs) or set(froze) != set(pairs):
                    raise RuntimeError(
                        "a fixed-slot compact prior failed its KL gate; refusing "
                        "to substitute a different head")
            else:
                while len(froze) < n_freeze and cursor < len(order):
                    take = order[cursor:cursor + (n_freeze - len(froze))]
                    cursor += len(take)
                    pairs = [(idx // self.n_head, idx % self.n_head) for idx in take]
                    froze += self._freeze_batch(pairs, means, T, causal)
        finally:
            self._phase_add("freeze_batch_total_s", freeze_t0)
            self._profile_selector_phases = False
        rates = [int(m.dyn_frozen.sum()) for m in self.mods]
        self.log(f"[freeze] iter {iter_num}: select_once signal={sig} froze {len(froze)} "
                 f"rate={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def _val_loss_fixed(self, batches, ctx):
        """Val loss over a FIXED list of batches (paired before/after -> low-noise Δ)."""
        was = self.model.training
        self.model.eval()
        tot = 0.0
        for X, Y in batches:
            with ctx:
                _, loss = self.model(X, Y)
            tot += float(loss)
        self.model.train(was)
        return tot / len(batches)

    def _freeze_batch(self, pairs, means, T, causal):
        """Freeze the given (li, h) pairs with the configured content; the single
        entry point every mode uses. Returns the pairs ACTUALLY frozen — with
        content='own_prior_decomposed', heads whose alpha+rho fit is poor
        (KL > decomp_max_kl) are skipped for this round (they may re-qualify at
        a later check as their pattern keeps stabilizing)."""
        pairs = list(pairs)
        if not pairs:
            return []
        decomp_contents = {
            "own_prior_decomposed", "random_decomposed", "uniform_decomposed"
        }
        if self.content not in decomp_contents:
            for li, h in pairs:
                self.mods[li].freeze_heads(
                    [h], self._pattern_for(li, h, means, T, causal).unsqueeze(0))
            return pairs
        if self.content in {"random_decomposed", "uniform_decomposed"}:
            n = len(pairs)
            if self.content == "random_decomposed":
                bank = self._random_decomp_banks.get(T)
                if bank is None:
                    shape = (self.total_heads, T)
                    bank = (
                        torch.randn(shape, generator=self.rng) * self.random_logit_std,
                        torch.randn(shape, generator=self.rng) * self.random_logit_std,
                    )
                    self._random_decomp_banks[T] = bank
                flat = torch.tensor(
                    [li * self.n_head + h for li, h in pairs], dtype=torch.long)
                alpha = bank[0].index_select(0, flat)
                rho = bank[1].index_select(0, flat)
            else:
                alpha = torch.zeros(n, T)
                rho = torch.zeros(n, T)
            by_layer = {}
            for index, (li, h) in enumerate(pairs):
                by_layer.setdefault(li, []).append((index, h))
            for li, entries in by_layer.items():
                indices = [index for index, _ in entries]
                heads = [h for _, h in entries]
                self.mods[li].freeze_heads_decomposed(
                    heads, alpha[indices], rho[indices])
            return pairs
        # Batched convex fit from the (c, d) sufficient statistics (per-head
        # loops are host-bound on busy nodes). The fit's autograd holds a few
        # (N, T, T) logit tensors alive per Adam step, so chunk the increment
        # to bound the transient workspace even at long context (at T=1024 a
        # chunk holds 128 heads: one chunk in practice).
        device = self.mods[0].dyn_frozen.device
        cz = causal.to(device)
        chunk = max(1, int(2 ** 27 // (T * T)))
        frozen_pairs, skipped = [], []
        for c0 in range(0, len(pairs), chunk):
            part = pairs[c0:c0 + chunk]
            stack_t0 = self._phase_clock() if self._profile_selector_phases else None
            P = torch.stack([means[li][h][:T, :T] for li, h in part]).to(device)
            if stack_t0 is not None:
                self._phase_add("mean_stack_transfer_s", stack_t0)
            stats_t0 = self._phase_clock() if self._profile_selector_phases else None
            c, d = decomp_stats(P, cz)
            if stats_t0 is not None:
                self._phase_add("sufficient_statistics_s", stats_t0)
            fit_t0 = self._phase_clock() if self._profile_selector_phases else None
            if self.decomp_fit_backend == "fft":
                alpha, rho, zf = fit_alpha_rho_from_stats_fft(
                    c, d, steps=self.decomp_fit_steps)
            else:
                alpha, rho, zf = fit_alpha_rho_from_stats(
                    c, d, cz, steps=self.decomp_fit_steps)
            if fit_t0 is not None:
                self._phase_add("alpha_rho_fit_s", fit_t0)
            # exact fit KL straight from the sufficient statistics — no (N,T,T)
            # softmax materialization:  KL = sum Pn ln Pn / T  +  CE, with
            # CE = (sum_i z_i - <c, alpha> - <d, rho>) / T
            install_t0 = self._phase_clock() if self._profile_selector_phases else None
            m = cz.bool()
            Pn = P.clamp_min(1e-9) * cz
            Pn = Pn / Pn.sum(-1, keepdim=True)
            negH = torch.where(m, Pn * Pn.clamp_min(1e-9).log(),
                               torch.zeros_like(Pn)).sum((-1, -2)) / T
            del P, Pn
            ce = (zf.sum(-1) - (c * alpha).sum(-1) - (d * rho).sum(-1)) / T
            kls = (negH + ce).tolist()               # one host sync per chunk
            by_layer = {}
            for j, (li, h) in enumerate(part):
                self.last_decomp_kl[(li, h)] = kls[j]
                if kls[j] > self.decomp_max_kl:
                    skipped.append((li, h, kls[j]))
                else:
                    by_layer.setdefault(li, []).append(j)
            for li, js in by_layer.items():          # one batched freeze/layer
                hs = [part[j][1] for j in js]
                self.mods[li].freeze_heads_decomposed(hs, alpha[js], rho[js],
                                                      z=zf[js])
                frozen_pairs += [(li, h) for h in hs]
            if install_t0 is not None:
                self._phase_add("kl_gate_and_install_s", install_t0)
                export_t0 = self._phase_clock()
                self.last_decomp_stats_chunks.append({
                    "pairs": torch.tensor(part, dtype=torch.int16),
                    "c": c.detach().float().cpu(),
                    "d": d.detach().float().cpu(),
                    "negative_entropy": negH.detach().float().cpu(),
                    "fit_kl": torch.tensor(kls, dtype=torch.float32),
                    "accepted": torch.tensor(
                        [kls[j] <= self.decomp_max_kl for j in range(len(part))],
                        dtype=torch.bool),
                })
                self._phase_add("statistics_export_s", export_t0)
        if skipped:
            self.decomp_skipped += len(skipped)
            det = " ".join(f"L{li}H{h}:{k:.3f}" for li, h, k in skipped)
            self.log(f"[freeze] decomposed fit skipped {len(skipped)} bad-fit head(s) "
                     f"(KL > {self.decomp_max_kl}): {det}")
        return frozen_pairs

    def _pattern_for(self, li, h, means, T, causal):
        if self.content == "own_prior":
            return means[li][h]
        if self.content == "own_prior_relative":
            return relative_attention_prior(means[li][h][:T, :T])
        if self.content == "own_prior_sharp":
            # softmax of the MEAN logit: a sharp, DETERMINISTIC prior (no sampling noise,
            # no broken within-row correlation). Fair "sharp" control vs the blurry mean:
            # unlike 'gaussian' it preserves softmax's shift-invariance, so it does not
            # manufacture attention-variance that a stable (low-variance) head doesn't have.
            logits = self._lm[li][h][:T, :T].masked_fill(causal == 0, float("-inf"))
            return torch.softmax(logits, dim=-1)
        if self.content == "own_prior_gaussian":
            # sample once (per-position, independently) from the head's logit distribution,
            # then softmax. NOTE: independent per-cell sampling breaks within-row correlation
            # and softmax shift-invariance -> injects noise the head doesn't really have; it
            # is a noisy, not a clean-sharp, prior. Prefer own_prior_sharp for the sharp comparison.
            lm = self._lm[li][h][:T, :T]
            ls = self._ls[li][h][:T, :T]
            logits = (lm + ls * torch.randn(T, T, generator=self.rng)).masked_fill(causal == 0, float("-inf"))
            return torch.softmax(logits, dim=-1)
        if self.content == "own_prior_dirichlet":
            pattern, concentration, deterministic = sample_dirichlet_pattern(
                means[li][h][:T, :T], self._attn_var[li][h][:T, :T], causal,
                rng=self.rng, scale=self.dirichlet_scale,
                min_concentration=self.dirichlet_min_concentration,
                max_concentration=self.dirichlet_max_concentration,
            )
            if self.last_dirichlet_concentration is None:
                self.last_dirichlet_concentration = torch.full(
                    (self.n_layer, self.n_head, T), float("nan"))
                self.last_dirichlet_fallback = torch.zeros(
                    (self.n_layer, self.n_head, T), dtype=torch.bool)
            self.last_dirichlet_concentration[li, h] = concentration
            self.last_dirichlet_fallback[li, h] = deterministic
            return pattern
        if self.content == "random":
            lg = torch.randn(T, T, generator=self.rng).masked_fill(causal == 0, float("-inf"))
            return torch.softmax(lg, dim=-1)
        return causal / causal.sum(-1, keepdim=True)  # uniform

    @torch.no_grad()
    def _freeze_per_layer(self, iter_num, get_batch, ctx):
        """Marcio v2: per-layer independent, small increment (<=1 head/layer/round),
        each commit gated by a PAIRED val-loss check (same val batches before/after) so
        the Δ is low-noise; hold a layer if its freeze would worsen val loss by > val_tol.
        No global cap -> layers diverge (some >25%, some ~0). Variance picks the candidate;
        val loss decides whether to keep it."""
        vbatches = [get_batch("val") for _ in range(self.val_batches)]
        phv, means = self._measure(get_batch, ctx)            # variance + priors
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        base = self._val_loss_fixed(vbatches, ctx)
        accepted = held = 0
        for li, m in enumerate(self.mods):
            unf = (~m.dyn_frozen).nonzero(as_tuple=True)[0].tolist()
            if not unf:
                continue
            h = min(unf, key=lambda hh: phv[li, hh].item())   # most-stable unfrozen head
            if not self._freeze_batch([(li, h)], means, T, causal):
                held += 1                                     # bad decomposed fit -> hold layer
                continue
            after = self._val_loss_fixed(vbatches, ctx)
            if after - base > self.val_tol:                   # would hurt -> hold this layer
                m.unfreeze_heads([h]); held += 1
            else:
                base = after; accepted += 1
        rates = [int(m.dyn_frozen.sum()) for m in self.mods]
        self.log(f"[freeze] iter {iter_num}: +{accepted} (held {held}) "
                 f"total={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def _freeze_significance(self, iter_num, get_batch, ctx):
        """Self-calibrating: proxy-select a small increment (lowest-variance heads), freeze it,
        and KEEP it only if the val-loss increase is NOT statistically significant vs the model's
        own batch-to-batch noise (paired t on per-batch Δ). Threshold is a unitless t -> transfers
        across models/sizes (no absolute nats). Held rounds retry later as more heads converge."""
        phv, means = self._measure(get_batch, ctx)
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        vbatches = [get_batch("val") for _ in range(self.val_batches)]

        def per_batch_loss():
            was = self.model.training
            self.model.eval()
            out = []
            for X, Y in vbatches:
                with ctx:
                    _, l = self.model(X, Y)
                out.append(float(l))
            self.model.train(was)
            return torch.tensor(out)

        before = per_batch_loss()
        frozen = torch.stack([m.dyn_frozen for m in self.mods])
        cand = [(phv[li, h].item(), li, h) for li in range(self.n_layer)
                for h in range(self.n_head) if not bool(frozen[li, h])]
        if not cand:
            return
        cand.sort()
        inc = cand[:max(1, int(round(0.02 * self.total_heads)))]   # ~2% increment
        froze = self._freeze_batch([(li, h) for _, li, h in inc], means, T, causal)
        if not froze:                                              # all bad-fit: retry later
            return
        d = per_batch_loss() - before                              # per-batch Δ
        mean_d = d.mean().item()
        se = (d.std(unbiased=True) / (len(d) ** 0.5)).item() if len(d) > 1 else 1e9
        t = mean_d / (se + 1e-9)
        if t >= self.sig_t:                                        # significant harm -> hold, retry later
            for li, h in froze:
                self.mods[li].unfreeze_heads([h])
            self.log(f"[freeze] iter {iter_num}: HELD (t={t:.2f}>={self.sig_t}, Δ={mean_d:+.4f}) "
                     f"rate={self.current_rate():.3f}")
        else:
            self.log(f"[freeze] iter {iter_num}: froze +{len(froze)} (t={t:.2f}, Δ={mean_d:+.4f}) "
                     f"rate={self.current_rate():.3f}")

    @torch.no_grad()
    def _freeze_threshold(self, iter_num, get_batch, ctx):
        """Mode B (faithful to Marcio): per-layer thresholds advanced by a val-loss FEEDBACK.
        Each round tentatively advances every layer's threshold one notch (freeze that layer's
        next most-stable head, a small increment); ONE paired val check gates the whole round.
        If the round's advance worsens val loss by > val_tol, REVERT it (keep the current rate);
        else keep. Variance orders each layer's candidates; val loss controls whether the rate
        advances. No global cap -> per-layer rates diverge."""
        vbatches = [get_batch("val") for _ in range(self.val_batches)]
        phv, means = self._measure(get_batch, ctx)
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        v_before = self._val_loss_fixed(vbatches, ctx)
        # each layer nominates its next most-stable head; advance only the globally
        # lowest-variance ~2% of them this round (small increment, per-layer-aware)
        cand = []
        for li, m in enumerate(self.mods):
            unf = (~m.dyn_frozen).nonzero(as_tuple=True)[0].tolist()
            if unf:
                h = min(unf, key=lambda hh: phv[li, hh].item())
                cand.append((phv[li, h].item(), li, h))
        cand.sort()
        round_cap = max(1, int(round(0.02 * self.total_heads)))
        newly = self._freeze_batch([(li, h) for _, li, h in cand[:round_cap]],
                                   means, T, causal)
        if not newly:
            return
        delta = self._val_loss_fixed(vbatches, ctx) - v_before
        if delta > self.val_tol:                              # advance hurt -> keep current rate
            for li, h in newly:
                self.mods[li].unfreeze_heads([h])
            self.log(f"[freeze] iter {iter_num}: HELD rate (val +{delta:+.4f} > {self.val_tol}) "
                     f"rate={self.current_rate():.3f}")
        else:
            rates = [int(m.dyn_frozen.sum()) for m in self.mods]
            self.log(f"[freeze] iter {iter_num}: advanced +{len(newly)} (val Δ={delta:+.4f}) "
                     f"rate={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def _freeze_relative(self, iter_num, get_batch, ctx):
        """Effect-size stopping rule — the portable, self-calibrating 'how many'.
        Freeze the globally lowest-variance heads in small (~2%) increments, and keep
        advancing only while the CUMULATIVE validation-perplexity increase caused by
        freezing stays <= val_rel (unitless, e.g. 0.01 = 1% PPL). Each round's marginal
        damage is measured on PAIRED val batches (same batches before/after -> low-noise
        Δ in nats) and accumulated; the budget in nats is log(1 + val_rel), so
        PPL_frozen / PPL_full <= 1 + val_rel. Unlike mode='significance' (zero-tolerance,
        froze 0) this PERMITS the small unavoidable per-freeze cost; unlike absolute
        val_tol (nats) it transfers across model sizes. When the next increment would
        break the budget it HOLDS (rolls back, retries later as heads keep converging).
        NOTE: the paired Δ is the *instantaneous* freeze cost; V re-adapts afterwards, so
        this is a conservative (upper-bound) estimate of the final cost."""
        budget = math.log1p(self.val_rel)                     # nats; PPL may rise <= val_rel
        vbatches = [get_batch("val") for _ in range(self.val_batches)]
        phv, means = self._measure(get_batch, ctx)
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        frozen = torch.stack([m.dyn_frozen for m in self.mods])
        cand = [(phv[li, h].item(), li, h) for li in range(self.n_layer)
                for h in range(self.n_head) if not bool(frozen[li, h])]
        if not cand:
            return
        cand.sort()
        # cap the round so we never overshoot the max_rate safety cap either
        room = int(self.max_rate * self.total_heads) - int(frozen.sum())
        n_inc = min(max(1, int(round(0.02 * self.total_heads))), max(room, 0))
        if n_inc <= 0:
            return
        inc = cand[:n_inc]                                    # ~2% lowest-variance increment
        v_before = self._val_loss_fixed(vbatches, ctx)
        froze = self._freeze_batch([(li, h) for _, li, h in inc], means, T, causal)
        if not froze:                                         # all bad-fit: retry later
            return
        delta = self._val_loss_fixed(vbatches, ctx) - v_before   # paired marginal damage (nats)
        if self.cum_damage + delta > budget:                  # would break budget -> hold, retry later
            for li, h in froze:
                self.mods[li].unfreeze_heads([h])
            self.log(f"[freeze] iter {iter_num}: HELD +{len(froze)} "
                     f"(Δ={delta:+.4f}, cum {self.cum_damage:.4f}+Δ > budget {budget:.4f} "
                     f"[≤{self.val_rel * 100:.2f}% PPL]) rate={self.current_rate():.3f}")
            return
        self.cum_damage += delta
        rates = [int(m.dyn_frozen.sum()) for m in self.mods]
        self.log(f"[freeze] iter {iter_num}: froze +{len(froze)} "
                 f"(Δ={delta:+.4f}, cum {self.cum_damage:.4f}/{budget:.4f} "
                 f"[cumPPL≈{math.expm1(self.cum_damage) * 100:+.2f}%]) "
                 f"rate={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def _freeze_ramp(self, iter_num, max_iters, get_batch, ctx):
        """CHEAP gradual freezing — NO val-loss probing, NO per-head trial-and-error.
        Freeze the lowest-<signal> heads on a linear schedule from 0 at warmup to max_rate
        at the end of training, so heads consolidate progressively (Version A). The ONLY
        per-round cost is one variance/prior measurement (`_measure`); with
        select_signal='gradient' the ranking uses the free grad-norm EMA and needs no extra
        forward at all. Designed to add negligible training latency (unlike the val-gated
        modes, whose O(T^2) val passes become prohibitive at long context)."""
        warm = int(self.warmup_frac * max_iters)
        end = int(self.ramp_end * max_iters)                             # freezing completes here
        frac = min(1.0, max(0.0, (iter_num - warm) / max(end - warm, 1)))
        target = int(round(self.max_rate * self.total_heads * frac))     # scheduled count
        have = sum(int(m.dyn_frozen.sum()) for m in self.mods)
        n_new = target - have
        if n_new <= 0:
            return
        phv, means = self._measure(get_batch, ctx)                       # prior content (+ variance)
        frozen = torch.stack([m.dyn_frozen for m in self.mods]).flatten()
        if self.select_signal == "gradient" and self.grad_ema is not None:
            key, sig = self.grad_ema.flatten(), "gradient"
        else:
            key, sig = phv.flatten(), "variance"
        if self.fixed_head_order:
            cand = [
                (float(rank), index)
                for rank, index in enumerate(self.fixed_head_order)
                if not bool(frozen[index])
            ]
            sig = "fixed_head_order"
        else:
            cand = sorted(
                (key[i].item(), i)
                for i in range(self.total_heads) if not bool(frozen[i]))
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        pairs = [(idx // self.n_head, idx % self.n_head) for _, idx in cand[:n_new]]
        froze = self._freeze_batch(pairs, means, T, causal)
        if (self.fixed_head_order
                and (len(froze) != len(pairs) or set(froze) != set(pairs))):
            raise RuntimeError(
                "a fixed-slot ramp prior failed its KL gate; refusing to "
                "substitute a different head")
        rates = [int(m.dyn_frozen.sum()) for m in self.mods]
        self.log(f"[freeze] iter {iter_num}: ramp({sig}) target={target} +{len(froze)} "
                 f"rate={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def observe_loss(self, iter_num, max_iters, x, get_batch, ctx):
        """Per-step hook for mode='plateau': a train-loss-stabilization-gated AUTO controller.
        Uses ONLY the free per-step train loss `x` (no extra forward; one `_measure` per freeze
        event). Self-calibrates a correlation time S from a self-terminating burn-in, then freezes
        one small increment of the lowest-signal (variance/grad) converged-tail heads on each fresh,
        power-verified train-loss PLATEAU (smoothed slope within its own noise); waits (ADAPT) until
        the value path re-absorbs the freeze — the fast EMA tracks the no-freeze COUNTERFACTUAL —
        before re-arming; rolls back a freeze that stays above the counterfactual (harm, from train
        loss alone); and stops when the converged tail is exhausted / on repeated harm / when too
        little training remains to adapt. When/how-many/when-to-stop all EMERGE — no warmup, no
        interval, no target rate, no loss-magnitude threshold anywhere."""
        if self.mode != "plateau" or self.pl_stopped or x is None or not self.mods:
            return
        z = self.pl_z

        # (A) short burn-in: seed the EMAs; spans are fractions of the run (dimensionless,
        # auto-scaling estimator settings — like grad_beta). S = the fast half-life.
        if self.pl_S is None:
            self.pl_buf.append(x)
            if len(self.pl_buf) >= max(32, int(round(0.01 * max_iters))):
                hf = max(8, int(round(0.005 * max_iters)))       # fast EMA half-life ~0.5% of run
                self.pl_S = hf
                self.pl_af = 1 - 2 ** (-1.0 / hf)
                self.pl_as = 1 - 2 ** (-1.0 / (self.pl_rho * hf))
                self.pl_av = self.pl_af
                m = sum(self.pl_buf) / len(self.pl_buf)
                self.pl_f = self.pl_s = m
                # seed noise var from DETRENDED diffs (var(noise) ~ var(diff)/2), NOT the
                # trend-inflated level variance -> a sane initial plateau band.
                d = [self.pl_buf[i] - self.pl_buf[i - 1] for i in range(1, len(self.pl_buf))]
                md = sum(d) / len(d)
                self.pl_v = max(sum((q - md) ** 2 for q in d) / len(d) / 2.0, 1e-12)
                self.pl_state = "WATCH"
                self.log(f"[freeze] plateau: burn-in done @{iter_num} half-life={hf} "
                         f"fast={self.pl_af:.3f} slow={self.pl_as:.3f}")
            return
        S = self.pl_S

        # (B) online fast/slow/noise EMAs + descent rate + early-rate reference
        af, as_, av = self.pl_af, self.pl_as, self.pl_av
        f_prev = self.pl_f
        self.pl_f = (1 - af) * self.pl_f + af * x
        self.pl_s = (1 - as_) * self.pl_s + as_ * x
        self.pl_v = (1 - av) * self.pl_v + av * (x - self.pl_f) ** 2
        self.pl_drate = (1 - as_) * self.pl_drate + as_ * (f_prev - self.pl_f)   # >0 improving
        sigma = max(self.pl_v, 0.0) ** 0.5
        n_t = sigma * (af / (2 - af)) ** 0.5             # slope-noise scale (self-calibrated band)
        g_t = self.pl_s - self.pl_f                      # >0 while loss falls, ->0 at plateau
        if g_t > self.pl_g_ref:
            self.pl_g_ref, self.pl_g_ref_since = g_t, 0
        else:
            self.pl_g_ref_since += 1

        # (C) ADAPT: enforce re-adaptation; counterfactual harm test (credits ongoing LR descent)
        if self.pl_state == "ADAPT":
            dt = iter_num - self.pl_t_freeze
            expected = self.pl_f_before - self.pl_drate_freeze * dt   # no-freeze descent
            gap = self.pl_f - expected
            if gap <= z * n_t:                            # tracked the counterfactual -> absorbed
                self.pl_adapt_times.append(dt)
                self.pl_state, self.pl_armed = "WATCH", False
                self.pl_g_ref, self.pl_g_ref_since = 0.0, 0
            elif dt > 4 * S:                              # never recovered -> the freeze hurt
                for li, h in self.pl_last_inc:
                    self.mods[li].unfreeze_heads([h])
                self.pl_harm_count += 1
                self.pl_bar = self.pl_last_bar_min        # only strictly lower-signal heads henceforth
                self.log(f"[freeze] plateau @{iter_num}: HARM rollback +{len(self.pl_last_inc)} "
                         f"(gap={gap:+.4f}>{z * n_t:.4f}) harm={self.pl_harm_count} "
                         f"rate={self.current_rate():.3f}")
                self.pl_state, self.pl_armed = "WATCH", False
                self.pl_g_ref, self.pl_g_ref_since = 0.0, 0
                if self.pl_harm_count >= self.pl_harm_max:
                    self.pl_stopped = True
            return

        # (D) WATCH: freeze on a fresh, power-verified plateau
        plateau = g_t < z * n_t
        power_ok = self.pl_g_ref > 0 and n_t <= self.pl_g_ref / z   # can resolve early rate at z-sigma
        peak_passed = self.pl_g_ref_since >= S
        if not self.pl_started:
            if not (peak_passed and power_ok and plateau):
                return
            self.pl_started, self.pl_armed = True, True    # arm the first freeze
        if not self.pl_armed:                              # re-arm latch: descent must resume first
            if g_t > z * n_t:
                self.pl_armed = True
            return
        if not plateau:
            return
        # late-freeze guard: need enough steps left to adapt (uses MEASURED adaptation times)
        mean_adapt = (sum(self.pl_adapt_times) / len(self.pl_adapt_times)
                      if self.pl_adapt_times else self.pl_rho * S)
        if (max_iters - iter_num) < mean_adapt or self.current_rate() >= self.max_rate:
            self.pl_stopped = True
            self.log(f"[freeze] plateau @{iter_num}: STOP (late-guard/cap) rate={self.current_rate():.3f}")
            return
        # freeze one self-sized increment of the converged (lowest-signal) tail
        phv, means = self._measure(get_batch, ctx)
        frozen = torch.stack([m.dyn_frozen for m in self.mods]).flatten()
        if self.select_signal == "gradient" and self.grad_ema is not None:
            key = self.grad_ema.flatten()
        else:
            key = phv.flatten()
        unf = sorted((key[i].item(), i) for i in range(self.total_heads) if not bool(frozen[i]))
        if not unf:
            self.pl_stopped = True
            return
        # freeze the lowest-signal unfrozen heads; the ONLY gate is pl_bar (=inf until a harm
        # event tightens it to strictly-lower-signal). Rate self-limits via harm detection +
        # late-guard + max_rate, NOT a preset percentile. (A Tukey lower fence was wrong here:
        # a low-side outlier bar is almost always below the minimum -> nothing eligible.)
        eligible = [(v, i) for v, i in unf if v < self.pl_bar]
        if not eligible:                                   # harm raised the bar below all remaining -> stop
            self.pl_stopped = True
            self.log(f"[freeze] plateau @{iter_num}: STOP (converged tail exhausted) "
                     f"rate={self.current_rate():.3f}")
            return
        cap = max(1, int(round(self.pl_inc_cap * self.total_heads)))
        inc = eligible[:cap]
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        pairs = [(idx // self.n_head, idx % self.n_head) for _, idx in inc]
        self.pl_last_inc = self._freeze_batch(pairs, means, T, causal)
        if not self.pl_last_inc:
            return                      # whole increment was bad-fit: stay in WATCH, retry
        self.pl_last_bar_min = min(v for v, _ in inc)
        self.pl_f_before, self.pl_drate_freeze, self.pl_t_freeze = self.pl_f, max(self.pl_drate, 0.0), iter_num
        self.pl_state, self.pl_armed = "ADAPT", False
        rates = [int(m.dyn_frozen.sum()) for m in self.mods]
        self.log(f"[freeze] plateau @{iter_num}: froze +{len(self.pl_last_inc)} "
                 f"(signal<={inc[-1][0]:.3g} g={g_t:.4g} "
                 f"n={n_t:.4g}) rate={self.current_rate():.3f} per-layer={rates}")

    @torch.no_grad()
    def maybe_freeze(self, iter_num, max_iters, get_batch, ctx):
        if self.mode == "plateau":
            return   # plateau mode is driven per-step by observe_loss()
        if not self.mods or iter_num % self.M != 0:
            return
        warm = int(self.warmup_frac * max_iters)
        if iter_num < warm or self.current_rate() >= self.max_rate:
            return

        if self.mode == "per_layer":
            return self._freeze_per_layer(iter_num, get_batch, ctx)
        if self.mode == "threshold":
            return self._freeze_threshold(iter_num, get_batch, ctx)
        if self.mode == "select_once":
            return self._freeze_select_once(iter_num, get_batch, ctx)
        if self.mode == "significance":
            return self._freeze_significance(iter_num, get_batch, ctx)
        if self.mode == "relative":
            return self._freeze_relative(iter_num, get_batch, ctx)
        if self.mode == "ramp":
            return self._freeze_ramp(iter_num, max_iters, get_batch, ctx)

        phv, means = self._measure(get_batch, ctx)
        frozen = torch.stack([m.dyn_frozen for m in self.mods])  # (n_layer, n_head)

        if self.tau_lo is None:  # calibrate the threshold band once
            allv = phv.flatten()
            self.tau_lo = torch.quantile(allv, self.p_lo / 100.0).item()
            self.tau_hi = torch.quantile(allv, self.p_hi / 100.0).item()

        prog = min(1.0, (iter_num - warm) / max(max_iters - warm, 1))
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * prog  # relax over training

        cand = [(phv[li, h].item(), li, h)
                for li in range(self.n_layer) for h in range(self.n_head)
                if not bool(frozen[li, h]) and phv[li, h].item() < tau]
        cand.sort()
        budget = int(self.max_rate * self.total_heads) - int(frozen.sum())
        cand = cand[:max(budget, 0)]
        if not cand:
            self.log(f"[freeze] iter {iter_num}: tau={tau:.4g} -> no new heads "
                     f"(rate={self.current_rate():.3f})")
            return

        # val loss BEFORE this round's freeze (for the guard)
        vb = self._val_loss(get_batch, ctx) if self.val_tol > 0 else None

        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        froze = self._freeze_batch([(li, h) for _, li, h in cand], means, T, causal)
        if not froze:
            self.log(f"[freeze] iter {iter_num}: tau={tau:.4g} -> no new heads "
                     f"(all bad-fit) rate={self.current_rate():.3f}")
            return

        # val-loss guard (Marcio): roll back if total damage would exceed the budget
        if self.val_tol > 0:
            va = self._val_loss(get_batch, ctx)
            delta = va - vb                              # marginal damage of this round
            if self.cum_damage + delta > self.val_tol:
                for li, h in froze:
                    self.mods[li].unfreeze_heads([h])
                self.log(f"[freeze] iter {iter_num}: HELD {len(froze)} heads "
                         f"(val +{delta:+.4f}, cum {self.cum_damage:.4f}, budget {self.val_tol}) "
                         f"rate={self.current_rate():.3f}")
                return
            self.cum_damage += delta
            self.log(f"[freeze] iter {iter_num}: froze {len(froze)} heads "
                     f"val Δ={delta:+.4f} cum={self.cum_damage:.4f} -> rate={self.current_rate():.3f}")
            return

        self.log(f"[freeze] iter {iter_num}: tau={tau:.4g} froze {len(froze)} heads "
                 f"-> rate={self.current_rate():.3f}")

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

import torch

# the canonical decomposed-logits definition lives next to the kernel
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from kernel.fused_attn import decomp_logits  # noqa: E402


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
        batch_mean.detach().double().cpu(),
        batch_m2.detach().double().cpu(),
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
                 seed=42, log=None):
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
        self._lm = self._ls = None   # per-position logit mean/std (for the gaussian prior)
        self._attn_var = None        # true per-sample attention variance (for Dirichlet)
        self.last_dirichlet_concentration = None
        self.last_dirichlet_fallback = None
        # content='own_prior_decomposed': frozen priors are stored as alpha+rho
        # logit VECTORS (O(T)/head, fitted from the (c, d) sufficient statistics).
        # Heads whose fit KL exceeds decomp_max_kl are SKIPPED for the round
        # (not dense-frozen — that would silently reintroduce T^2 storage).
        repr_ = getattr(self.mods[0], "prior_repr", "dense") if self.mods else "dense"
        if content == "own_prior_decomposed" and repr_ != "decomposed":
            raise ValueError("content='own_prior_decomposed' requires the model to be "
                             "built with rand_attn_prior_repr='decomposed'")
        if repr_ == "decomposed" and content != "own_prior_decomposed":
            raise ValueError(
                f"rand_attn_prior_repr='decomposed' with freeze_content='{content}' "
                "would freeze DENSE T^2 patterns and lazily allocate the very "
                "buffer the decomposed mode exists to avoid; set "
                "freeze_content='own_prior_decomposed'")
        self.decomp_max_kl = decomp_max_kl
        self.decomp_fit_steps = decomp_fit_steps
        self.decomp_skipped = 0      # cumulative count of bad-fit skips (logged)
        self.last_decomp_kl = {}     # (li, h) -> fit KL of the most recent fit
        self.sig_t = sig_t           # t-threshold for significance-gated freezing (unitless)
        self.select_signal = select_signal   # 'variance' | 'gradient' | 'combined'
        self.grad_beta = grad_beta
        self.grad_ema = None     # EMA of per-head Q/K grad-norm (convergence signal)
        self.rng = torch.Generator().manual_seed(seed)
        self.log = log or (lambda *a, **k: print(*a, flush=True, **k))
        self.tau_lo = self.tau_hi = None
        self.n_layer = len(self.mods)
        self.n_head = int(self.mods[0].dyn_frozen.numel()) if self.mods else 0
        self.total_heads = self.n_layer * self.n_head

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

    @torch.no_grad()
    def _measure(self, get_batch, ctx):
        """Run K batches with capture on; return per-head variance (n_layer,n_head)
        and each layer's mean attention (n_head,T,T).

        ``variance_estimator='sample'`` computes the actual variance across all
        K*B examples. ``batch_mean`` preserves the historical estimator (variance
        across K batch means) for exact reproduction of older experiments.
        """
        was_training = self.model.training
        self.model.eval()
        for m in self.mods:
            m.set_capture(True)
        att_mean = [None] * self.n_layer   # float64 CPU Chan/Welford statistics
        att_m2 = [None] * self.n_layer
        logit_mean = [None] * self.n_layer
        logit_m2 = [None] * self.n_layer
        legacy_mean = [None] * self.n_layer
        legacy_m2 = [None] * self.n_layer
        n_samp = 0
        n_logit = 0
        n_batches = 0
        for _ in range(self.K):
            X, Y = get_batch("train")
            with ctx:
                self.model(X, Y)
            n_batches += 1
            for li, m in enumerate(self.mods):
                att = m._last_attn.float()                       # (B, n_head, T, T)
                bn, bm, b2 = _centered_batch_moments(att)
                count = n_samp if att_mean[li] is not None else 0
                _, att_mean[li], att_m2[li] = _merge_moments(
                    count, att_mean[li], att_m2[li], bn, bm, b2)
                # Historical estimator: variance across equal-weight batch means.
                _, legacy_mean[li], legacy_m2[li] = _merge_moments(
                    n_batches - 1, legacy_mean[li], legacy_m2[li],
                    1, bm, torch.zeros_like(bm))
                lg = m._last_logits.float()                    # (B, n_head, T, T) raw logits
                lbn, lbm, lb2 = _centered_batch_moments(lg)
                lcount = n_logit if logit_mean[li] is not None else 0
                _, logit_mean[li], logit_m2[li] = _merge_moments(
                    lcount, logit_mean[li], logit_m2[li], lbn, lbm, lb2)
            n_samp += bn
            n_logit += lbn
        for m in self.mods:
            m.set_capture(False)
        self.model.train(was_training)
        means = [att_mean[li].float() for li in range(self.n_layer)]
        sample_var = [
            (att_m2[li] / max(n_samp - 1, 1)).clamp_min(0).float()
            for li in range(self.n_layer)
        ]
        if self.variance_estimator == "sample":
            selected_var = sample_var
        else:
            selected_var = [
                (legacy_m2[li] / max(n_batches - 1, 1)).clamp_min(0).float()
                for li in range(self.n_layer)
            ]
        per_head_var = torch.stack([selected_var[li].mean(dim=(-1, -2))
                                    for li in range(self.n_layer)])
        self.last_var = per_head_var
        self._attn_var = sample_var
        # per-position logit mean / std across inputs (for the sampled 'gaussian' prior)
        self._lm = [logit_mean[li].float() for li in range(self.n_layer)]
        self._ls = [
            (logit_m2[li] / max(n_logit - 1, 1)).clamp_min(0).sqrt().float()
            for li in range(self.n_layer)
        ]
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
        variance = input-independence (a prior can represent it); gradient = convergence
        (safe to stop learning it); combined = low in BOTH (rank-sum)."""
        if self.current_rate() > 0:
            return
        phv, means = self._measure(get_batch, ctx)
        n_freeze = int(self.max_rate * self.total_heads)
        var = phv.flatten()
        if self.select_signal == "gradient" and self.grad_ema is not None:
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
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        # iso-rate contract: if decomposed fitting SKIPS bad-fit heads, top the
        # quota up from the next-lowest-signal candidates (this mode never runs
        # again, so skips would otherwise permanently undershoot the rate)
        froze, cursor = [], 0
        while len(froze) < n_freeze and cursor < len(order):
            take = order[cursor:cursor + (n_freeze - len(froze))]
            cursor += len(take)
            pairs = [(idx // self.n_head, idx % self.n_head) for idx in take]
            froze += self._freeze_batch(pairs, means, T, causal)
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
        if self.content != "own_prior_decomposed":
            for li, h in pairs:
                self.mods[li].freeze_heads(
                    [h], self._pattern_for(li, h, means, T, causal).unsqueeze(0))
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
            P = torch.stack([means[li][h][:T, :T] for li, h in part]).to(device)
            c, d = decomp_stats(P, cz)
            alpha, rho, zf = fit_alpha_rho_from_stats(c, d, cz,
                                                      steps=self.decomp_fit_steps)
            # exact fit KL straight from the sufficient statistics — no (N,T,T)
            # softmax materialization:  KL = sum Pn ln Pn / T  +  CE, with
            # CE = (sum_i z_i - <c, alpha> - <d, rho>) / T
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
        if skipped:
            self.decomp_skipped += len(skipped)
            det = " ".join(f"L{li}H{h}:{k:.3f}" for li, h, k in skipped)
            self.log(f"[freeze] decomposed fit skipped {len(skipped)} bad-fit head(s) "
                     f"(KL > {self.decomp_max_kl}): {det}")
        return frozen_pairs

    def _pattern_for(self, li, h, means, T, causal):
        if self.content == "own_prior":
            return means[li][h]
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
        cand = sorted((key[i].item(), i) for i in range(self.total_heads) if not bool(frozen[i]))
        T = int(means[0].size(-1))
        causal = torch.tril(torch.ones(T, T))
        pairs = [(idx // self.n_head, idx % self.n_head) for _, idx in cand[:n_new]]
        froze = self._freeze_batch(pairs, means, T, causal)
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

"""
Unit tests for the decomposed-prior controller pieces (dynamic_freeze.py):

  1. SUFFICIENT-STATISTICS FIT: fitting alpha+rho from (c, d) only must reach
     the same solution as the full-matrix cross-entropy fit (the objective
     couples to P through c and d alone — gradients are identical).
  2. CONTROLLER INTEGRATION: with content='own_prior_decomposed', a freeze
     round on a tiny dynamic model must (a) freeze heads via
     freeze_heads_decomposed (dyn_decomp set, no dyn_pattern allocated),
     (b) leave the cheap path matching the eager oracle, and (c) SKIP every
     head when decomp_max_kl is set impossibly low.
  3. MATCHED CONTROLS: random/uniform decomposed priors freeze the same quota,
     retain O(T) storage, and produce causal row-stochastic patterns.
  4. RANDOM-BANK CONTROL: a head receives exactly the same structured-random
     alpha/rho vectors across rates, pair orders, and incremental calls.

Run:  python nanogpt/test_decomposed_controller.py
"""
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT                                     # noqa: E402
from dynamic_freeze import (DynamicFreezeController, decomp_stats,   # noqa: E402
                            fit_alpha_rho_from_stats, decomp_materialize_q,
                            fit_alpha_rho_from_stats_fft,
                            decomp_log_normalizers_fft, decomp_logits,
                            kl_rows_batched, parse_fixed_head_order)
from fit_prior_decomposition import fit_alpha_rho_batched            # noqa: E402


def attn_modules(model):
    mods = []
    for blk in model.transformer.h:
        m = blk.main_block
        if hasattr(m, "block"):
            m = m.block
        mods.append(m)
    return mods


def test_stats_fit(device):
    """stats fit == matrix fit (same convergence point of the convex MLE)."""
    torch.manual_seed(0)
    N, T = 6, 256
    causal = torch.tril(torch.ones(T, T, device=device))
    # ground-truth decomposable patterns + a noise perturbation in logit space
    a0 = torch.randn(N, T, device=device) * 0.8
    r0 = -torch.arange(T, device=device).float() / 10.0 + torch.randn(N, T, device=device) * 0.1
    P = decomp_materialize_q(a0, r0, causal)
    P = P + 0.02 * torch.rand_like(P) * causal            # break exact decomposability
    P = P / P.sum(-1, keepdim=True)

    c, d = decomp_stats(P, causal)
    a_s, r_s, z_s = fit_alpha_rho_from_stats(c, d, causal, steps=600)
    a_f, r_f, z_f = fit_alpha_rho_from_stats_fft(c, d, steps=600)
    Q_s = decomp_materialize_q(a_s, r_s, causal)
    Q_f = decomp_materialize_q(a_f, r_f, causal)
    Q_m, _, _ = fit_alpha_rho_batched(P, causal, steps=600)

    kl_s = kl_rows_batched(P, Q_s, causal)
    kl_m = kl_rows_batched(P, Q_m, causal)
    gap = (kl_s - kl_m).abs().max().item()
    qdiff = (Q_s - Q_m).abs().max().item()
    # the controller's gate computes this same KL from the statistics alone
    # (no materialized Q): KL = sum Pn ln Pn / T + (sum z - <c,a> - <d,r>) / T
    m = causal.bool()
    Pn = P.clamp_min(1e-9) * causal
    Pn = Pn / Pn.sum(-1, keepdim=True)
    negH = torch.where(m, Pn * Pn.clamp_min(1e-9).log(),
                       torch.zeros_like(Pn)).sum((-1, -2)) / T
    kl_stats = negH + (z_s.sum(-1) - (c * a_s).sum(-1) - (d * r_s).sum(-1)) / T
    sgap = (kl_stats - kl_s).abs().max().item()
    fft_z_gap = (decomp_log_normalizers_fft(a_s, r_s) - z_s).abs().max().item()
    fft_kl_gap = (kl_rows_batched(P, Q_f, causal) - kl_s).abs().max().item()
    fft_q_gap = (Q_f - Q_s).abs().max().item()
    # The FFT normalizer must also preserve the dense objective's gradient.
    ag = a0.detach().clone().requires_grad_(True)
    rg = r0.detach().clone().requires_grad_(True)
    dense_objective = torch.logsumexp(decomp_logits(ag, rg), dim=-1).sum()
    dense_grad = torch.autograd.grad(dense_objective, (ag, rg))
    af = a0.detach().clone().requires_grad_(True)
    rf = r0.detach().clone().requires_grad_(True)
    fft_objective = decomp_log_normalizers_fft(af, rf).sum()
    fft_grad = torch.autograd.grad(fft_objective, (af, rf))
    fft_grad_gap = max(
        (dense_grad[0] - fft_grad[0]).abs().max().item(),
        (dense_grad[1] - fft_grad[1]).abs().max().item())
    # the claim under test is EQUIVALENCE: same convergence point from stats
    # alone (the absolute KL just reflects how decomposable the synthetic P is)
    ok = (gap < 1e-3 and qdiff < 5e-3 and sgap < 1e-4
          and fft_z_gap < 2e-4 and fft_kl_gap < 1e-3 and fft_q_gap < 5e-3
          and fft_grad_gap < 2e-4)
    print(f"  [stats-fit] KL(stats) mean={kl_s.mean():.4f} max={kl_s.max():.4f}  "
          f"|KL gap| vs matrix-fit={gap:.2e}  max|Q_s-Q_m|={qdiff:.2e}  "
          f"stats-KL vs materialized-KL |gap|={sgap:.2e}  "
          f"FFT z/KL/Q/grad gaps={fft_z_gap:.2e}/{fft_kl_gap:.2e}/"
          f"{fft_q_gap:.2e}/{fft_grad_gap:.2e}  "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def make_model_and_batch(device, repr_):
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=128,
                    vocab_size=512, dropout=0.0, bias=False,
                    rand_attn_dynamic=True, rand_attn_prior_repr=repr_)
    model = GPT(cfg).to(device).eval()
    def get_batch(split):
        x = torch.randint(0, 512, (2, 128), device=device)
        return x, x
    return model, get_batch


def test_controller(device, max_kl, expect_frozen):
    torch.manual_seed(1)
    model, get_batch = make_model_and_batch(device, "decomposed")
    ctrl = DynamicFreezeController(
        model, content="own_prior_decomposed", mode="select_once",
        check_interval=1, warmup_frac=0.0, max_rate=0.5, measure_batches=2,
        decomp_max_kl=max_kl, decomp_fit_steps=300,
        log=lambda *a, **k: None)
    ctrl.maybe_freeze(1, 10, get_batch, contextlib.nullcontext())
    mods = attn_modules(model)
    n_fz = sum(int(m.dyn_frozen.sum()) for m in mods)
    n_dec = sum(int(m.dyn_decomp.sum()) for m in mods)
    no_patt = all(m.dyn_pattern is None for m in mods)
    if expect_frozen:
        ok = n_fz > 0 and n_dec == n_fz and no_patt
    else:
        ok = n_fz == 0 and ctrl.decomp_skipped > 0
    expected_phases = {
        "capture_and_mean_s", "ranking_s", "freeze_batch_total_s",
        "mean_stack_transfer_s", "sufficient_statistics_s",
        "alpha_rho_fit_s", "kl_gate_and_install_s", "statistics_export_s",
    }
    profiled = expected_phases.issubset(ctrl.last_selector_phases)
    reusable_stats = bool(ctrl.last_decomp_stats_chunks) and all(
        chunk["c"].shape == chunk["d"].shape
        and chunk["c"].shape[0] == chunk["pairs"].shape[0]
        for chunk in ctrl.last_decomp_stats_chunks)
    ok &= profiled and reusable_stats
    # cheap path must still match the eager oracle after (possible) freezing
    x, y = get_batch("train")
    with torch.no_grad():
        for m in mods:
            m.set_capture(True)
        ye, _ = model(x, y)
        for m in mods:
            m.set_capture(False)
        yc, _ = model(x, y)
    diff = (ye - yc).abs().max().item()
    ok &= diff < 1e-3
    print(f"  [controller max_kl={max_kl}] frozen={n_fz} decomposed={n_dec} "
          f"skipped={ctrl.decomp_skipped} dyn_pattern_none={no_patt} "
          f"profiled={profiled} reusable_stats={reusable_stats} "
          f"oracle |Δ|={diff:.1e} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_decomposed_control(device, content):
    torch.manual_seed(7)
    model, get_batch = make_model_and_batch(device, "decomposed")
    ctrl = DynamicFreezeController(
        model, content=content, mode="select_once", check_interval=1,
        warmup_frac=0.0, max_rate=0.5, measure_batches=2, seed=123,
        log=lambda *a, **k: None)
    ctrl.maybe_freeze(1, 10, get_batch, contextlib.nullcontext())
    mods = attn_modules(model)
    n_frozen = sum(int(module.dyn_frozen.sum()) for module in mods)
    no_dense = all(module.dyn_pattern is None for module in mods)
    stochastic = False
    max_row_error = 0.0
    for module in mods:
        idx = module.dyn_frozen
        if not bool(idx.any()):
            continue
        patterns = module._decomp_patterns(idx, 128)
        max_row_error = max(max_row_error,
                            float((patterns.sum(-1) - 1).abs().max()))
        stochastic |= bool(module.dyn_alpha[idx].abs().max() > 0)
    expected_stochastic = content == "random_decomposed"
    ok = (n_frozen == 4 and no_dense and max_row_error < 1e-4
          and stochastic == expected_stochastic)
    print(f"  [{content}] frozen={n_frozen} no_dense={no_dense} "
          f"row_error={max_row_error:.1e} stochastic={stochastic} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def test_fixed_head_order(device):
    requested = "L1H3,L0H2,L1H0,L0H1"
    parsed = parse_fixed_head_order(requested, n_layer=2, n_head=4)
    parser_ok = parsed == [7, 2, 4, 1]
    try:
        parse_fixed_head_order("L0H1,L0H1", n_layer=2, n_head=4)
        parser_ok = False
    except ValueError:
        pass
    try:
        parse_fixed_head_order("L2H0", n_layer=2, n_head=4)
        parser_ok = False
    except ValueError:
        pass

    torch.manual_seed(11)
    model, get_batch = make_model_and_batch(device, "decomposed")
    ctrl = DynamicFreezeController(
        model, content="uniform_decomposed", mode="select_once",
        check_interval=1, warmup_frac=0.0, max_rate=0.5,
        measure_batches=1, fixed_head_order=requested,
        log=lambda *a, **k: None)
    ctrl.maybe_freeze(1, 10, get_batch, contextlib.nullcontext())
    actual = [
        li * 4 + head
        for li, module in enumerate(attn_modules(model))
        for head in range(4) if bool(module.dyn_frozen[head])
    ]
    ok = parser_ok and set(actual) == set(parsed) and len(actual) == len(parsed)

    # The fitted compact installer groups writes by layer. Its return order is
    # allowed to differ from the cross-layer request order, but its slot set
    # must remain exact.
    fitted_model, fitted_batch = make_model_and_batch(device, "decomposed")
    fitted_ctrl = DynamicFreezeController(
        fitted_model, content="own_prior_decomposed", mode="select_once",
        check_interval=1, warmup_frac=0.0, max_rate=0.5,
        measure_batches=1, fixed_head_order=requested, decomp_max_kl=10.0,
        decomp_fit_steps=2, log=lambda *a, **k: None)
    fitted_ctrl.maybe_freeze(1, 10, fitted_batch, contextlib.nullcontext())
    fitted_actual = [
        li * 4 + head
        for li, module in enumerate(attn_modules(fitted_model))
        for head in range(4) if bool(module.dyn_frozen[head])
    ]
    ok &= set(fitted_actual) == set(parsed) and len(fitted_actual) == len(parsed)

    # The Phase 1A dense oracle resumes from a decomposed-capable unfrozen
    # checkpoint, then explicitly opts into the lazy dense diagnostic buffer.
    dense_model, dense_batch = make_model_and_batch(device, "decomposed")
    dense_ctrl = DynamicFreezeController(
        dense_model, content="own_prior", mode="select_once",
        check_interval=1, warmup_frac=0.0, max_rate=0.5,
        measure_batches=1, fixed_head_order=requested,
        allow_dense_fallback=True, log=lambda *a, **k: None)
    dense_ctrl.maybe_freeze(1, 10, dense_batch, contextlib.nullcontext())
    dense_modules = attn_modules(dense_model)
    dense_actual = [
        li * 4 + head
        for li, module in enumerate(dense_modules)
        for head in range(4) if bool(module.dyn_frozen[head])
    ]
    dense_ok = (
        set(dense_actual) == set(parsed)
        and all(module.dyn_pattern is not None for module in dense_modules)
    )
    ok &= dense_ok
    print(f"  [fixed-order] requested={parsed} actual={actual} "
          f"fitted_actual={fitted_actual} "
          f"dense_actual={dense_actual} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_random_bank_invariance(device):
    model_a, _ = make_model_and_batch(device, "decomposed")
    model_b, _ = make_model_and_batch(device, "decomposed")
    ctrl_a = DynamicFreezeController(
        model_a, content="random_decomposed", mode="select_once", seed=314,
        log=lambda *a, **k: None)
    ctrl_b = DynamicFreezeController(
        model_b, content="random_decomposed", mode="select_once", seed=314,
        log=lambda *a, **k: None)
    T = 128
    causal = torch.tril(torch.ones(T, T))
    shared = [(0, 0), (1, 2)]
    ctrl_a._freeze_batch(shared, [], T, causal)
    # Different order and a larger set model a different frozen rate.
    ctrl_b._freeze_batch([(0, 3), (1, 2), (0, 0), (1, 1)], [], T, causal)
    # A later incremental call must read from the same persistent bank.
    ctrl_a._freeze_batch([(0, 3)], [], T, causal)

    mods_a, mods_b = attn_modules(model_a), attn_modules(model_b)
    exact = True
    for li, h in shared + [(0, 3)]:
        exact &= torch.equal(mods_a[li].dyn_alpha[h].cpu(),
                             mods_b[li].dyn_alpha[h].cpu())
        exact &= torch.equal(mods_a[li].dyn_rho[h].cpu(),
                             mods_b[li].dyn_rho[h].cpu())
    print(f"  [random-bank] shared heads invariant across rate/order/calls: "
          f"{'OK' if exact else 'FAIL'}")
    return exact


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    ok = test_stats_fit(device)
    ok &= test_controller(device, max_kl=10.0, expect_frozen=True)
    ok &= test_controller(device, max_kl=1e-6, expect_frozen=False)
    ok &= test_decomposed_control(device, "random_decomposed")
    ok &= test_decomposed_control(device, "uniform_decomposed")
    ok &= test_fixed_head_order(device)
    ok &= test_random_bank_invariance(device)
    print(f"\n=> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

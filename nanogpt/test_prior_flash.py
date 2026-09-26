"""
End-to-end integration test for the prior-load + Flash-compatibility flow.

This is THE test to run while working on the efficient kernel. It exercises the full
path the kernel must support:

    load attention priors from disk  ->  freeze the chosen heads to those priors
    ->  forward pass  ->  (1) cheap skip-QK path must match the eager reference
    ->  (2) the fused mixed-head Triton kernel must actually dispatch (GPU, bf16).

Run on a CUDA GPU to actually exercise Flash; on CPU it still checks correctness
(SDPA falls back to the math backend, and the Flash assertion is reported as N/A).

    python nanogpt/test_prior_flash.py                       # synthetic realistic prior, fresh 124M
    python nanogpt/test_prior_flash.py --prior_path prior.pt # load a real extracted prior
    python nanogpt/test_prior_flash.py --ckpt ckpt.pt        # load trained weights too
    python nanogpt/test_prior_flash.py --T 2048 --rate 0.25  # longer context / freeze rate

Prior file format (`--prior_path`): a torch-saved float tensor of shape
(n_layer, n_head, T, T), causal and row-normalised (this is what `extract_attention.py`
produces, averaged over a corpus). If omitted, a realistic local-decay prior is
generated so the test runs out of the box.
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as model_mod                                      # noqa: E402
from model import GPTConfig, GPT                                 # noqa: E402

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAVE_SDPA_CTX = True
except Exception:
    HAVE_SDPA_CTX = False


def attn_modules(model):
    mods = []
    for blk in model.transformer.h:
        m = blk.main_block
        if hasattr(m, "block"):
            m = m.block
        mods.append(m)
    return mods


def make_realistic_prior(n_layer, n_head, T):
    """A causal, row-normalised local-decay prior (resembles a real attention prior:
    mostly attend to nearby tokens, with a per-head decay scale)."""
    i = torch.arange(T).view(T, 1).float()
    j = torch.arange(T).view(1, T).float()
    dist = i - j                                  # >=0 on/under the diagonal
    P = torch.empty(n_layer, n_head, T, T)
    for l in range(n_layer):
        for h in range(n_head):
            scale = 2.0 + 3.0 * ((l * n_head + h) % 5)
            w = torch.exp(-dist / scale).masked_fill(dist < 0, 0.0)
            P[l, h] = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_path", default=None, help="(n_layer,n_head,T,T) prior tensor")
    ap.add_argument("--ckpt", default=None, help="optional trained checkpoint (loaded strict=False)")
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--rate", type=float, default=0.25, help="fraction of heads/layer to freeze")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--position_encoding", choices=("learned_absolute", "rope"),
                    default="learned_absolute")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.n_layer}L/{args.n_head}H/{args.n_embd}d  T={args.T}  rate={args.rate}")

    cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    block_size=args.T, vocab_size=50304, dropout=0.0, bias=False,
                    rand_attn_dynamic=True,
                    position_encoding=args.position_encoding)
    model = GPT(cfg).to(device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = sd.get("model", sd)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})")

    # 1) LOAD PRIOR
    if args.prior_path:
        obj = torch.load(args.prior_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            # extract_attention.py saves {"mean": (L,H,T,T), "var": ..., "samples": ...}
            prior = obj.get("mean", obj.get("prior"))
            assert prior is not None, "prior dict must contain a 'mean' (or 'prior') key"
        else:
            prior = obj
        print(f"  loaded prior {tuple(prior.shape)} from {args.prior_path}")
    else:
        prior = make_realistic_prior(args.n_layer, args.n_head, args.T)
        print(f"  generated realistic prior {tuple(prior.shape)}")
    assert prior.shape == (args.n_layer, args.n_head, args.T, args.T), \
        f"prior must be (n_layer,n_head,T,T)={(args.n_layer,args.n_head,args.T,args.T)}, got {tuple(prior.shape)}"

    # 2) FREEZE the chosen heads to the loaded prior
    k = max(1, int(round(args.rate * args.n_head)))
    for li, a in enumerate(attn_modules(model)):
        idx = list(range(k))                         # freeze the first k heads of each layer
        a.freeze_heads(idx, prior[li, idx].to(device))
    frozen = [int(a.dyn_frozen.sum()) for a in attn_modules(model)]
    print(f"  frozen per layer: {frozen}  (total {sum(frozen)}/{args.n_layer*args.n_head})")

    x = torch.randint(0, 50304, (args.batch, args.T), device=device)

    def set_capture(flag):
        for a in attn_modules(model):
            a.set_capture(flag)

    # 3a) CORRECTNESS in fp32 (no autocast). The eager oracle computes attention
    # manually while the cheap path uses SDPA for unfrozen heads; those two kernels
    # only agree numerically in fp32. (In bf16 they legitimately differ ~1e-2 — that
    # is kernel noise, not a regression. Validate math in fp32.)
    with torch.no_grad():
        set_capture(True)
        ye, _ = model(x, x)
        set_capture(False)
        yc, _ = model(x, x)
    diff = (ye.float() - yc.float()).abs().max().item()
    match = diff < 1e-3
    print(f"  [forward fp32] cheap vs eager max|Δ| = {diff:.2e}  ->  {'MATCH' if match else 'MISMATCH'}")

    # 3b) Dispatch assertion in bf16. The fused path implements its own
    # Flash-style online softmax and therefore does NOT call PyTorch SDPA;
    # verify the fused call directly. The legacy split path still verifies SDPA.
    flash_ok = None
    if device == "cuda" and HAVE_SDPA_CTX:
        if model_mod._FUSED:
            calls = 0
            original = model_mod.fused_mixed_attn
            def counted(*a, **kw):
                nonlocal calls
                calls += 1
                return original(*a, **kw)
            model_mod.fused_mixed_attn = counted
            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
                flash_ok = calls == args.n_layer
            except Exception as e:
                flash_ok = False
                print(f"    fused error: {type(e).__name__}: {e}")
            finally:
                model_mod.fused_mixed_attn = original
            print(f"  [fused bf16] mixed-head Triton calls: {calls}/{args.n_layer} "
                  f"-> {'OK' if flash_ok else 'FAILED'}")
        else:
            try:
                with torch.no_grad(), sdpa_kernel([SDPBackend.FLASH_ATTENTION]), \
                     torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
                flash_ok = True
            except Exception as e:
                flash_ok = False
                print(f"    flash error: {type(e).__name__}: {e}")
            print(f"  [split bf16] unfrozen heads on PyTorch Flash SDPA: "
                  f"{'OK' if flash_ok else 'FAILED'}")
    else:
        print("  [flash] N/A on CPU (run on a CUDA GPU to exercise Flash)")

    ok = match and (flash_ok is not False)
    print(f"\n=> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

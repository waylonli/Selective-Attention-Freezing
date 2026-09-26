import json
from pathlib import Path

import pytest
import torch

from nanogpt.model import GPT, GPTConfig
from nanogpt.eval_nanogpt_logprobs import load_model
from scripts.export_checkpoint import evaluation_payload, export
from scripts.pretrain import command
from supplement.attention import install_packed


@pytest.mark.parametrize("mode", ["ordinary", "prune", "uniform", "mean"])
def test_checkpoint_roundtrip(tmp_path, mode):
    torch.manual_seed(17)
    config = GPTConfig(block_size=64, vocab_size=32, n_layer=1, n_head=2, n_embd=16,
                       dropout=0., rand_attn_dynamic=True, rand_attn_prior_repr="decomposed",
                       position_encoding="rope")
    model = GPT(config).eval()
    heads = [[0]] if mode != "ordinary" else [[]]
    if mode in ("prune", "uniform"):
        install_packed(model, heads, mode)
    elif mode == "mean":
        attn = model.transformer.h[0].main_block
        attn.dyn_frozen[0] = True
        attn.dyn_decomp[0] = True
        attn.refresh_dynamic_derived_state(rebuild_rho_band=True)
    source = tmp_path / "source.pt"
    payload = {"model": model.state_dict(), "model_config": vars(config),
               "optimizer": {"private": "not exported"},
               "intervention": {"mode": mode, "heads": heads}}
    torch.save(payload, source)
    metadata = export(source, tmp_path / "export")
    restored, _, saved = load_model(tmp_path / "export/model.pt", "cpu")
    x = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
    assert "optimizer" not in saved
    assert metadata["strict_load_passed"]
    torch.load(tmp_path / "export/model.pt", weights_only=True)


def test_removes_private_task_records():
    payload = evaluation_payload({"model_args": {}, "model": {},
                                  "task_finetune": {"task": "boolq", "official_validation_examples": ["text"]}})
    assert payload["task_finetune"] == {"task": "boolq"}


def test_manifest_commands():
    root = Path(__file__).resolve().parents[1]
    for manifest in (root / "configs/pretraining").glob("*.json"):
        for arm in json.loads(manifest.read_text())["arms"]:
            argv = command(arm, Path("outputs/test"))
            assert (root / arm["config"]).is_file()
            assert any(arg.startswith("--out_dir=") for arg in argv)


def test_no_private_assets():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Source hygiene checks require a Git checkout")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in filter(None, names):
        path = root / name
        assert path.suffix not in {".csv", ".pt", ".pth", ".bin", ".sbatch", ".pdf"}
        text = path.read_text()
        assert "/" + "projects/" not in text
        assert "/" + "Users/" not in text


def test_mqar_generator_determinism_and_rng_isolation():
    from scripts.mqar import data
    state = torch.get_rng_state()
    x, y = data(800521, 512, 8, 4)
    assert torch.equal(state, torch.get_rng_state())
    a, b = data(800521, 512, 8, 4)
    assert torch.equal(a, x) and torch.equal(b, y)
    assert ((y != -100).sum(1) == 8).all()


def test_cpu_training_entry_point(tmp_path):
    import subprocess
    import sys
    import numpy as np
    import pickle
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.arange(1024, dtype=np.uint16) % 32
    tokens.tofile(data / "train.bin")
    tokens.tofile(data / "val.bin")
    with (data / "meta.pkl").open("wb") as stream:
        pickle.dump({"vocab_size": 128, "stoi": {chr(i): i for i in range(128)},
                     "itos": {i: chr(i) for i in range(128)}}, stream)
    settings = dict(dataset=str(data), out_dir=str(tmp_path / "model"), debug_data=False,
                    n_layer=1, n_head=2, n_embd=16, block_size=64, seq_len=64,
                    batch_size=2, gradient_accumulation_steps=1, max_iters=2,
                    lr_decay_iters=2, warmup_iters=0, eval_interval=1, eval_iters=1,
                    device="cpu", dtype="float32", compile=False, wandb_log=False,
                    rand_attn_dynamic=True, rand_attn_prior_repr="decomposed",
                    freeze_max_rate=0.0, always_save_checkpoint=True)
    result = subprocess.run([sys.executable, "nanogpt/train.py",
                             *[f"--{k}={v!r}" for k, v in settings.items()]],
                            cwd=root, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (tmp_path / "model/ckpt.pt").exists()

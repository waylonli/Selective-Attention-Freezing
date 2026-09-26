# Selective Attention Freezing (SAF)

**When Can Attention Heads Be Statically Defined?**

**Weixian Waylon Li**, **Yintao Tai**, **Marcio Fonseca** and **Shay B. Cohen**.

SAF replaces selected attention heads with fixed causal patterns while retaining their input-dependent value projections and token mixing.
The repository contains native-model training and evaluation (`nanogpt/`), fused kernels (`kernel/`), experiment configurations (`configs/`), controls (`supplement/`) and pretrained Qwen evaluation (`qwen/`).

## Installation

Use Python 3.10 or newer and a PyTorch build appropriate for your machine.
The recorded GPU environment used PyTorch 2.10 and Triton 3.6; fused execution requires Linux and an NVIDIA GPU with BF16 support.

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest
```

Run commands from the repository root unless stated otherwise.
Use `--help` on each entry point for its full options.
Corpora, weights and generated results are not included.

## Released Assets

[Checkpoints](https://huggingface.co/waylonli/Selective-Attention-Freezing) and [evaluation data and splits](https://huggingface.co/datasets/waylonli/Selective-Attention-Freezing-data) are hosted on Hugging Face.
The data release contains the original 124M and 1B validation-token files, nine downstream task/seed partitions, and six MQAR pair-sweep datasets.
Checkpoint uploads are ongoing; the [live catalogue](https://huggingface.co/waylonli/Selective-Attention-Freezing/blob/main/checkpoints.json) marks available entries as `uploaded`.
The download commands use immutable revisions and verify SHA256 checksums:

```bash
python -m scripts.download --data --out data/release
python -m scripts.download --checkpoint 124m-t16384-s1337-saf25 --out checkpoints
```

The checkpoint command prints the downloaded `model.pt` path.
Choose other IDs from the catalogue for ordinary attention, 1B, task-finetuned models and controls.
Each model directory includes its configuration, export checksums and validation metadata.
Exports preserve tensor precision, fixed patterns and pruning layouts, but omit optimiser state.

## Data and Training

Native models use the GPT-2 tokeniser through `tiktoken` and `uint16` token files.
The 124M preparation script is `data/fineweb_edu/prepare.py`; its archived training-corpus revision remains to be confirmed, so use the released original validation file for exact evaluation.
The 1B data are revision-pinned, with expected token-file hashes in `configs/data_1b.json`:

```bash
python data/prepare_fineweb_edu_scale.py \
  --out-dir data/fineweb_edu_20b --train-tokens 20100000000 \
  --validation-tokens 10000000 \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

python -m scripts.pretrain --manifest configs/pretraining/main_v1.json --list
python -m scripts.pretrain --manifest configs/pretraining/main_v1.json \
  --run-id p0-baseline-124m-c4096-rope-s1337 --dry-run
```

Remove `--dry-run` to train; use `--out` for a new output directory and `--parent` for a continuation checkpoint containing Adam, sampler and RNG state.
Evaluation-only exports cannot resume the original training trajectory.

| Experiment | Configuration in `configs/pretraining/` |
| --- | --- |
| Initial trajectories, rates and times | `main_v1.json` |
| Representation and pattern content | `main_v2_phase1a.json`, `main_v2_phase1b*.json` |
| Head selection and schedules | `main_v2_phase1c.json`, `main_v2_phase1d*.json`, `kl_selector_continuation_v1.json` |
| Three-seed 4K recipe | `main_v2_phase2_replication.json` plus seed 1337 in `main_v1.json` |
| Native 8K and 16K | `main_v2_phase4*.json`, `main_v2_phase5*.json` |
| 1B | `scaleup_1b_midpoint_v1.json`, `scaleup_1b_continuations_v1.json`, `scaleup_1b_frozen_retry_v2.json` |

Use `frozen_retry_v2` for the final 1B SAF models and `continuations_v1` for ordinary attention.
The 1B study uses seed 1337, 8K context and four GPUs; the 124M recipe uses seeds 1337--1339.

For matched-token/time controls, run `python -m supplement.train --help` with `configs/controls.json`.
Clock arms require the ordinary-attention summary and midpoint receipt; keep independent token/time cohorts separate.
Add `--learn-patterns` to a mean arm for the trainable alpha/rho comparison, using the same parent and head set.
The maturity study uses `python -m supplement.maturity_train` and the dependency graph in `configs/maturity/manifest.json`, with a separate data partition.
The four-GPU 1B pruning entry point is `python -m scripts.prune_1b`; it requires the full 18,756-update ordinary checkpoint.

## Evaluation and Adaptation

Use native nanoGPT checkpoints, not Transformers `AutoModel` files:

```bash
python nanogpt/eval_nanogpt_logprobs.py \
  --ckpt checkpoints/124m-pretraining/124m-t16384-s1337-saf25/model.pt \
  --data_dir data/release/pretraining/fineweb_edu \
  --start_fraction 0.5 --end_fraction 1.0 --batch_size 1 \
  --out outputs/logprobs.pt

python -m scripts.eval_task \
  --ckpt checkpoints/finetuned/model.pt --task boolq --max-length 512 \
  --out outputs/boolq.json

python -m scripts.mqar eval \
  --ckpt checkpoints/mqar/model.pt --length 512 --pairs 8 16 24 32 48 64 \
  --out outputs/mqar
```

Use task-adapted checkpoints for downstream and MQAR accuracy, not their pretraining parents.
For SST-2, BoolQ and QuALITY training, use `nanogpt/finetune_downstream.py` with `--save_ckpt`.
Run ordinary attention first with `--min_selected_epoch 1`, then pass its internally selected epoch to paired alternatives with `--fixed_epochs`; do not select on official validation results.
QuALITY uses native 16K checkpoints at 124M and 8K checkpoints at 1B.

`python -m scripts.mqar train --ckpt checkpoints/parent/model.pt --out outputs/mqar-adapted` adapts on eight pairs at 512 tokens, with batch 16, learning rate 1e-4 and 1,500 updates.
The main pair sweep uses later-intervention matched-time parents and three pretraining seeds; keep midpoint and task-seed controls separate.

## Speed Benchmarks

```bash
python nanogpt/bench_checkpoint_training.py \
  --baseline-ckpt checkpoints/ordinary/model.pt --frozen-ckpt checkpoints/saf/model.pt \
  --T 4096 --B 8 --grad-accum 15 --tokens-per-update 491520

python nanogpt/bench_checkpoint_prefill.py \
  --baseline-ckpt checkpoints/ordinary16k/model.pt --frozen-ckpt checkpoints/saf16k/model.pt \
  --T 4096 --B 64 --warmup 20 --iters 80 --repeats 6
```

Use matching checkpoints, dtypes and GPU allocations.
The prefill length sweep uses native 16K checkpoints and batch 64; the batch sweep fixes length at 4K.
CPU reference execution does not measure the reported acceleration.

## Pretrained Qwen

Run from `qwen/` to use its separate GQA implementation (source revision `9b9dc79`):

```bash
cd qwen
pip install -r requirements.txt
python nanogpt/prior_corpus_matrix.py --model Qwen/Qwen3-4B \
  --corpora fineweb_edu --no_mixed --seq_len 2048 --batch_size 2 \
  --extract_batches 256 --eval_batches 64 --placement kl_guided \
  --rates 0,0.1,0.2,0.3 --out_dir outputs/priors
python nanogpt/fit_for_selector.py --ckpt_dir outputs/priors --signal phv --rate 0.3
python nanogpt/downstream_eval.py --model Qwen/Qwen3-4B \
  --ckpt_dir outputs/priors --placement variance_guided --rate 0.1 \
  --out outputs/tasks-variance10.json
```

Repeat evaluation at rates 0, 0.2 and 0.3, and with `--placement kl_guided`.
The downstream protocol uses 512 calibration sequences, not the separate 128-sequence prior set.

## Weights and Tests

`checkpoints.json` is a local snapshot of the release catalogue; the downloader uses the current Hugging Face catalogue.
Export a trusted training checkpoint without changing tensor precision or dropping fixed-pattern buffers and pruning layouts:

```bash
python -m scripts.export_checkpoint --source /path/to/training.pt \
  --out /path/to/hf-staging/model-name --expected-sha256 SOURCE_HASH --trust-source

python -m pytest tests supplement/test_controls.py nanogpt/test_data_sampling.py
python -m pytest nanogpt/test_distributed_freeze.py
python kernel/test_fused_attn.py
```

The distributed test needs local loopback access (`GLOO_SOCKET_IFNAME=lo0` on macOS if required); kernel tests need CUDA.
Published checkpoints pass strict loading and bitwise original/exported BF16-logit checks on CUDA; their `metadata.json` files record the validation scope.
Kernel correctness tests are separate from export checks.

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT), with its MIT licence and attribution retained in [LICENSE](LICENSE).
MQAR uses [Zoology revision `1ad20d1`](https://github.com/HazyResearch/zoology/tree/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb), covered by [zoology/LICENSE](zoology/LICENSE).
Its generation function is unchanged; this distribution replaces the configuration framework with a minimal return container.
Datasets and third-party weights remain subject to their upstream terms.

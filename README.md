# Selective Attention Freezing (SAF)

SAF replaces selected attention heads with fixed causal patterns while retaining their input-dependent value projections and token mixing.
This repository contains the training, calibration, compact fitting, fused execution and evaluation code used in the paper.

## Installation

Use Python 3.10 or newer and install a CUDA-enabled PyTorch build appropriate for your machine.
The recorded GPU environment used PyTorch 2.10 and Triton 3.6; CPU tests also run with PyTorch 2.9.

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest
```

Fused benchmarks require Linux, an NVIDIA GPU with BF16 support, and Triton.
CPU execution provides reference implementations, not the reported acceleration.

## Layout

| Directory | Contents |
| --- | --- |
| `nanogpt/` | Model, training, head selection, fitting, task finetuning and paired benchmarks |
| `kernel/` | Fused mixed-head attention and indexed projections |
| `scripts/` | Portable training launcher, checkpoint export, task evaluation and MQAR |
| `configs/` | Scientific configurations, without scheduler or account settings |
| `supplement/` | Matched-budget controls, pruning and trainable-pattern comparison |
| `qwen/` | Pretrained Qwen GQA extraction and zero-shot evaluation |
| `zoology/` | Vendored MQAR generator and upstream licence |

[REPRODUCING.md](REPRODUCING.md) gives commands and identifies the relevant configurations.
[CHECKPOINTS.md](CHECKPOINTS.md) describes evaluation weights and the Hugging Face release inventory.
[THIRD_PARTY.md](THIRD_PARTY.md) records upstream code and dataset sources.

No training corpora, model weights, prediction dumps or historical result archives are included.
Weights have not yet been uploaded: `checkpoints.json` is an inventory, not a list of available downloads.

## Quick Evaluation

Run from the repository root, using an exported model checkpoint:

```bash
python nanogpt/eval_nanogpt_logprobs.py \
  --ckpt checkpoints/model.pt --data_dir data/fineweb_edu \
  --start_fraction 0.5 --end_fraction 1.0 --batch_size 1 \
  --out outputs/logprobs.pt

python -m scripts.eval_task \
  --ckpt checkpoints/finetuned/model.pt --task boolq --max-length 512 \
  --out outputs/boolq.json

python -m scripts.mqar eval \
  --ckpt checkpoints/mqar/model.pt --length 512 --pairs 8 16 24 32 48 64 \
  --out outputs/mqar
```

Use task-adapted checkpoints for finetuning and MQAR accuracy, not their pretraining parents.
The checkpoint format is native nanoGPT, not a Transformers `AutoModel` format.

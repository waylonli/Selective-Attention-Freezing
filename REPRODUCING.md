# Reproducing the Experiments

Commands below run from the repository root unless stated otherwise.
Use separate output directories for every model, seed and budget.
Generated results are deliberately excluded from version control.

## Data

All native models use the GPT-2 tokeniser through `tiktoken` and `uint16` token files.
`data/fineweb_edu/prepare.py` provides the original 124M preparation procedure.
For exact numerical evaluation, check the published token-file hashes; a newly downloaded corpus can differ from the archived data.
The original 124M corpus revision still needs to be recovered from the training archive before the weight release.

The 1B data preparation is revision-pinned:

```bash
python data/prepare_fineweb_edu_scale.py \
  --out-dir data/fineweb_edu_20b --train-tokens 20100000000 \
  --validation-tokens 10000000 \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

`configs/data_1b.json` records the expected binary hashes and partition.
The maturity study uses a separate 22B-token stream, with a document-boundary split between source training and recovery.
Do not substitute its validation partition for the original pretraining partition.

## Native Pretraining

Configurations retain the original run identifiers so that parent relationships remain explicit.
The launcher can list a manifest, print its command, or run one arm without Slurm:

```bash
python -m scripts.pretrain --manifest configs/pretraining/main_v1.json --list
python -m scripts.pretrain --manifest configs/pretraining/main_v1.json \
  --run-id p0-baseline-124m-c4096-rope-s1337 --dry-run
```

Remove `--dry-run` to train.
Use `--out` to change the output directory and `--parent` to supply a corresponding full training checkpoint when branching.
An evaluation-only export does not contain the Adam or sampler state required for continuation training.

| Experiment | Configuration family |
| --- | --- |
| Initial trajectories, rates and times | `main_v1` |
| Dense versus compact representation | `main_v2_phase1a` |
| Pattern content | `main_v2_phase1b*` |
| Head selection and schedules | `main_v2_phase1c`, `main_v2_phase1d*`, `kl_selector_continuation_v1` |
| Three-seed 4K recipe | `main_v2_phase2_replication` plus seed 1337 in `main_v1` |
| Native 8K and 16K models | `main_v2_phase4*`, `main_v2_phase5*` |
| 1B midpoint and continuation | `scaleup_1b_midpoint_v1`, `scaleup_1b_continuations_v1`, `scaleup_1b_frozen_retry_v2` |

The final 1B SAF models use the `frozen_retry_v2` configurations, not the earlier frozen continuations.
The ordinary-attention continuation remains in `scaleup_1b_continuations_v1`.
The 1B study uses one pretraining seed, 8K context, four GPUs, 37,512 total updates and 19,667,091,456 training tokens.
The 124M recipe uses 5,000 updates of 491,520 tokens and three seeds, 1337--1339.
Changing the GPU count is supported when global accumulation remains divisible by the number of ranks.

## Matched-Budget Controls

```bash
python -m supplement.train --manifest configs/controls.json \
  --arm ordinary_tokens --seed 1337 --data data/fineweb_edu \
  --out outputs/controls/ordinary

python -m supplement.train --manifest configs/controls.json \
  --arm mean25_tokens --seed 1337 --data data/fineweb_edu \
  --parent outputs/controls/ordinary/midpoint.pt \
  --prefix-receipt outputs/controls/ordinary/midpoint.receipt.json \
  --out outputs/controls/mean25
```

For a clock arm, also pass `--ordinary-summary outputs/controls/ordinary/summary.json`.
The reported token and clock cohorts use independent ordinary-attention runs; keep those cohorts separate.
Calibration, fitting and intervention are included in the clock budget.
Use a separate run with `--learn-patterns` on a mean arm to continue learning the fitted alpha/rho parameters.
The frozen and trainable arms must start from the same parent and selected head set.

For the training-budget study, the manifest in `configs/maturity/` defines the phase graph:

```bash
python -m supplement.maturity_train bundle --source configs/maturity \
  --root outputs/maturity --name source_low
python -m supplement.maturity_train bundle --source configs/maturity \
  --root outputs/maturity --name recover_low_mean
```

It also defines variance and gate-Taylor pruning, matched-time recovery, and later source checkpoints.
Run dependencies first, as listed in each task's `needs` field.
For the single-seed 1B pruning comparison, `python -m scripts.prune_1b` takes `--parent`, `--expected-sha256`, `--data` and a new `--out` directory.
It verifies the native 1B token files, scores and physically packs the ordinary midpoint, preserves Adam state, then launches four-rank continuation.
The parent must be the full 18,756-update checkpoint, not an evaluation export.
GPU execution of this packaged launcher still needs validation before release.

## Downstream Finetuning

`nanogpt/finetune_downstream.py` implements SST-2, BoolQ and QuALITY with answer-token supervision.
Run the ordinary-attention baseline first, with `--min_selected_epoch 1`; it chooses an epoch using an internal training split.
Pass that integer to each paired alternative as `--fixed_epochs`.
Use `--save_ckpt` for the weights subsequently evaluated by `scripts.eval_task`.

```bash
python nanogpt/finetune_downstream.py --ckpt checkpoints/pretrained/model.pt \
  --task boolq --max_length 512 --epochs 5 --min_selected_epoch 1 \
  --seed 1337 --batch_size 16 --grad_accum 1 \
  --out outputs/boolq/result.json --save_ckpt outputs/boolq/finetuned.pt
```

Repeat with seeds 1338 and 1339.
Do not choose epochs on the official validation split.
QuALITY uses native 16K checkpoints for the 124M long-input experiment and native 8K checkpoints for 1B.
The large-batch systems benchmark is separate from the original quality-training configuration.

## Associative Recall

```bash
python -m scripts.mqar train --ckpt checkpoints/parent/model.pt \
  --seed 1337 --out outputs/mqar-adapted
```

The defaults match the pair-sweep adaptation: 512 tokens, eight pairs, batch 16, learning rate 1e-4 and 1,500 updates.
Evaluation varies the number of pairs without further training.
The main curve uses later-intervention matched-time parents and three pretraining seeds.
Task-seed replication and midpoint frozen/trainable comparisons use different parents and must not be pooled with that curve.
The upstream generation algorithm is unchanged; only its return container has been separated from the upstream training framework.

## Systems Benchmarks

```bash
python nanogpt/bench_checkpoint_training.py \
  --baseline-ckpt checkpoints/ordinary/model.pt --frozen-ckpt checkpoints/saf/model.pt \
  --T 4096 --B 8 --grad-accum 15 --tokens-per-update 491520

python nanogpt/bench_checkpoint_prefill.py \
  --baseline-ckpt checkpoints/ordinary16k/model.pt --frozen-ckpt checkpoints/saf16k/model.pt \
  --T 4096 --B 64 --warmup 20 --iters 80 --repeats 6
```

Both commands emit measurements to stdout.
Use matching checkpoints, batch sizes, dtypes and GPU allocations.
The prefill length sweep uses the same native 16K checkpoints and batch 64; the batch sweep fixes length at 4K.
These are prefill measurements, not decode or full distributed-training speedups.

## Tests

```bash
python -m pytest tests supplement/test_controls.py nanogpt/test_data_sampling.py
python -m pytest nanogpt/test_distributed_freeze.py
python kernel/test_fused_attn.py
```

The distributed CPU test needs permission to bind a local loopback socket.
On macOS, set `GLOO_SOCKET_IFNAME=lo0` if automatic interface selection fails.
CUDA numerical and timing tests must be rerun on the target GPU after packaging; they cannot be validated on a CPU-only host.

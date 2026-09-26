# Checkpoint Release

`checkpoints.json` lists evaluation-weight candidates, grouped by experiment.
It contains no server paths, prediction dumps or result tables.
`source_sha256` identifies an archived training file; it is not the hash of the smaller exported file.
Entries marked `needs_server_hash` must be checked against the source archive before upload.
No download repository has been assigned yet.

## Upload Groups

| Group | Models | Purpose |
| --- | ---: | --- |
| `124m-pretraining` | 27 | Ordinary attention, SAF 25% and SAF 50%; 4K/8K/16K; three pretraining seeds |
| `1b-pretraining` | 3 | Ordinary attention, SAF 25% and SAF 50%; 8K; seed 1337 |
| `124m-finetuning` | 27 | Three inherited variants, three tasks, three finetuning seeds |
| `1b-finetuning` | 27 | Same task/variant grid at 1B; one pretraining seed |
| `mqar-pair-sweep` | 12 | Four alternatives, three pretraining seeds; task-adapted weights |
| `matched-budget-controls` | 48 | Token/time comparisons, separate ordinary references and early-intervention control |
| `supplementary-controls` | 81 | Audited pruning, maturity, task-seed and trainable-pattern checkpoints |

The first five groups form a 96-checkpoint core evaluation release.
The two control groups add 129 candidates for the other reported comparisons; they are not additional SAF seeds.
The inventory is not yet exhaustive for every prior-content and schedule ablation in the appendix.
Those trajectories can be regenerated from the included configurations; their historical evaluation weights require a second archive pass if direct evaluation of every appendix cell is required.

For Qwen, retain the upstream `Qwen/Qwen3-4B` model rather than uploading repeated copies.
Release the 512-sequence calibration statistics and both selector-specific fitted pattern files used by the downstream experiments, together with the base-model revision, tokeniser revision, masks and extraction settings.
The 128-sequence corpus-transfer/prefill priors are a different configuration and must not replace them.

## Export

Export only trusted training files, outside the code repository:

```bash
python -m scripts.export_checkpoint --source /path/to/training.pt \
  --out /path/to/hf-staging/model-name --expected-sha256 SOURCE_HASH --trust-source
```

The export retains every model tensor and its original dtype, including fixed patterns and derived buffers.
It preserves physically pruned layouts and normalises the two historical checkpoint schemas.
It removes Adam state, RNG state, private paths, full task records and dataset examples.
Every export must strictly reload before its metadata is written.
Weights saved after training alpha/rho can be evaluated as fixed patterns; resuming their learning requires the trainable-pattern implementation and an appropriate training checkpoint.

Upload `model.pt`, `config.json`, `metadata.json`, and `attention_layout.json` when present, plus a model card.
The card should identify the parent, model size, context, replacement rate, selection rule, training and task seeds, task-training settings, intended evaluation command, tokeniser and chosen weight licence.
Use `tiktoken:gpt2` for native models; do not claim Transformers compatibility.
Do not cast FP32 weights to BF16 merely to reduce upload size when reproducing existing evaluation results.

Model-only weights cannot resume the original training trajectory.
Keep midpoint checkpoints with Adam, sampler and RNG state in a separate optional training-resume release.
Final optimiser states, timing-cache copies, predictions and intermediate milestones are not needed for evaluation uploads.

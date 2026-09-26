# Pretrained Qwen Evaluation

Run commands from this directory so its GQA kernel is loaded in a separate process from the native MHA implementation.
Install `requirements.txt` here for prior extraction; evaluation from prepared priors does not need the corpus streaming dependency.

For the reported downstream protocol, use `Qwen/Qwen3-4B`, 512 calibration sequences of 2,048 tokens, seed 42 and the FineWeb-Edu source.
Both selection criteria need fitted priors for all their selected heads.

```bash
python nanogpt/prior_corpus_matrix.py --model Qwen/Qwen3-4B \
  --corpora fineweb_edu --no_mixed --seq_len 2048 --batch_size 2 \
  --extract_batches 256 --eval_batches 64 --placement kl_guided \
  --rates 0,0.1,0.2,0.3 --out_dir outputs/priors

python nanogpt/fit_for_selector.py --ckpt_dir outputs/priors \
  --signal phv --rate 0.3

python nanogpt/downstream_eval.py --model Qwen/Qwen3-4B \
  --ckpt_dir outputs/priors --placement variance_guided --rate 0.1 \
  --out outputs/tasks-variance10.json
```

Repeat the final command at rates 0, 0.2 and 0.3, and with `--placement kl_guided`.
The task scorer uses exact-length buckets, avoiding padding in the causal fused path.
The extraction driver also evaluates held-out perplexity; these outputs are generated locally and are not bundled with the release.

For direct reproduction without extraction, obtain `stats_ckpt.pt`, `fit_fineweb_edu.pt` and `fit_fineweb_edu_phv.pt` from the authors' 512-sequence run.
Their exact hashes and the upstream model revision still need to be attached to the weight release.
Do not use the separate 128-sequence prior set for these downstream results.

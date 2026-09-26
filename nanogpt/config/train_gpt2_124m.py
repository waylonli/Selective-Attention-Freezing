# GPT-2 small (124M) config for the scaled-up prior experiments (two-stage / 3x3).
# Model: n_layer=12, n_head=12, n_embd=768, block_size==seq_len==1024 (steps==1).
# Batch: batch_size=24 x grad_accum=20 x seq_len=1024 = 491,520 tokens/iter
#        (the canonical ~0.5M-token GPT-2 batch; lr/warmup tuned for this regime).
#
# max_iters here is a placeholder for a ~1B-token run; the matrix/calibration
# runners OVERRIDE it from the chosen per-run token budget:
#   max_iters = round(target_tokens / 491520)   (e.g. 0.5B -> ~1017, 1B -> ~2035)
#
# Notes (verified against train.py / model.py):
#  - keep block_size == seq_len so the stateful chunk path is a single model() call.
#  - vocab_size auto-derives to 50304 (corpora are GPT-2 BPE tokenized).
#  - reported MFU is vs an A100 (312 TFLOP hardcoded); on GH200 (~990 TFLOP bf16)
#    the logged MFU is ~3.2x understated — use perf/iter_time_ms for GPU-hour math.

debug_data = False
dataset = 'fineweb_edu'   # overridden per run

eval_interval = 1000
eval_iters = 100
log_interval = 10
always_save_checkpoint = True   # need final weights for stage-1 -> extract_attention

wandb_log = False
wandb_project = 'nanogpt-124m-prior'

# batch: 24 x 20 x 1024 = 491,520 tokens/iter on a single 96GB GH200
gradient_accumulation_steps = 20
batch_size = 24
block_size = 1024
seq_len = 1024

# GPT-2 small
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
position_encoding = 'learned_absolute'  # override to 'rope' for the RoPE arm
rope_base = 10000.0

# optimizer (GPT-2 small schedule)
learning_rate = 6e-4
max_iters = 2035          # ~1B tokens; OVERRIDE per token budget
lr_decay_iters = 2035
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 200        # ~10% of a ~2k-iter run (scale with max_iters)

device = 'cuda'
dtype = 'bfloat16'
compile = True

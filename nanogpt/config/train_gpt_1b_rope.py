# Approximately 983M-parameter decoder used for the paper's scale confirmation.
# Four-rank DDP uses global gradient_accumulation_steps=64, which train.py
# divides to 16 local micro-steps: 4 ranks x B=1 x 16 x T=8192 = 524,288
# tokens per optimizer update. The readiness gate may use B=2/4 with global
# accumulation 32/16 while preserving exactly the same token batch.

dataset = "fineweb_edu_20b"
debug_data = False
preload_train = True
train_sampling = "shuffled_nonoverlap"
train_data_start_frac = 0.0
train_data_end_frac = 1.0
controller_val_start_frac = 0.0
controller_val_end_frac = 0.5
report_val_start_frac = 0.5
report_val_end_frac = 1.0

batch_size = 1
gradient_accumulation_steps = 64
block_size = 8192
seq_len = 8192

n_layer = 32
n_head = 16
n_embd = 1536
dropout = 0.0
bias = False
position_encoding = "rope"
rope_base = 10000.0

learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
max_iters = 37512
lr_decay_iters = 37512
warmup_iters = 375

eval_interval = 1875
eval_iters = 8
log_interval = 10
always_save_checkpoint = True
checkpoint_milestones = "18756"
wandb_log = False

# Keep the baseline architecture intervention-ready so its exact midpoint can
# branch into frozen continuations without changing model state structure.
rand_attn_dynamic = True
rand_attn_prior_repr = "decomposed"
freeze_content = "own_prior_decomposed"
freeze_mode = "select_once"
freeze_select_signal = "variance"
freeze_variance_estimator = "sample"
freeze_warmup_frac = 2.0
freeze_check_interval = 18756
freeze_max_rate = 0.0
freeze_measure_batches = 16
freeze_measure_batch_size = 1
freeze_decomp_fit_backend = "fft"
freeze_decomp_fit_steps = 400
freeze_decomp_max_kl = 0.2
freeze_pattern_dtype = "bfloat16"
freeze_allow_dense_fallback = False
freeze_intervention_eval_batches = 4

base_seed = 1337
rand_seed = 1337
normalize_gradient_accumulation = True
device = "cuda"
dtype = "bfloat16"
compile = False

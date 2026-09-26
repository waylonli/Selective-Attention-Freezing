"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import torch._dynamo
import os
import time
import math
import pickle
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import GPTConfig, GPT, fused_dispatch_count
from sample import load_tokenizer, encode_prompt, generate
from data_sampling import ShuffledNonoverlapSampler

torch._dynamo.config.suppress_errors = True

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
packed_attention_layout = None  # restored from physically pruned checkpoints only
skip_initial_eval_on_resume = False  # exact continuation: checkpoint was already evaluated/saved
checkpoint_milestones = ""  # comma-separated exact iteration snapshots, e.g. "1250,2000,2500"
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
debug_data = False # if True, use train_debug.bin / val_debug.bin subsets
# disjoint TRAIN data window (fraction of train.bin). Used for two-stage training:
# e.g. stage 1 trains on [0.0, 0.5), stage 2 on [0.5, 1.0) — fully disjoint tokens.
# Defaults span the whole file (no windowing). Applies to the 'train' split only;
# validation always uses the full val set.
train_data_start_frac = 0.0
train_data_end_frac = 1.0
# The paper protocol keeps scheduler decisions disjoint from final reporting.
# Defaults preserve historical runs; formal manifests override to [0,.5) and
# [.5,1) respectively.
controller_val_start_frac = 0.0
controller_val_end_frac = 1.0
report_val_start_frac = 0.0
report_val_end_frac = 1.0
# Load the whole train.bin into RAM once (np.fromfile) instead of memmap, to avoid
# per-batch random Lustre reads — important when many jobs run concurrently and the
# shared filesystem becomes the bottleneck. Needs RAM >= train.bin size.
preload_train = False
train_sampling = 'random_with_replacement'  # or 'shuffled_nonoverlap' for unique target tokens
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
normalize_gradient_accumulation = True  # divide each micro-batch gradient by accumulation count
                                        # (False reproduces the historical, over-scaled runs)
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
seq_len = 1024
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
state_size = 0
shared_weights = False
position_encoding = "learned_absolute"  # "learned_absolute" | "rope"
rope_base = 10000.0
# randomization controls
rand_head_rate = 0.0
rand_proj_scale = 1.0
rand_layer_rate = 0.0
rand_layer_mode = "evenly_spaced"
rand_layer_ids = ""
rand_components = "none"
rand_proj_mask_rate = 1.0
rand_attn_mask_rate = 0.0
rand_attn_head_rate = 0.0
rand_attn_head_rate_per_layer = ""   # e.g. "1.0,1.0,0.5,0.25,0.25,0.5" — overrides uniform rate if set
rand_attn_pattern = "none"
rand_attn_pattern_sharing = "none"   # "none" | "per_layer" | "global"
rand_attn_window = 32
rand_attn_prior_path = ""
rand_seed = 42
base_seed = 1337       # global torch seed (init + data order); vary for seed replicas
# dynamic per-head freezing (Version A: gradual attention training)
rand_attn_dynamic = False          # enable the variance-driven freezing controller
rand_attn_prior_repr = "dense"     # frozen-prior storage: 'dense' (T x T matrix) |
                                   # 'decomposed' (alpha[j]+rho[i-j] vectors, O(T)/head;
                                   # pair with freeze_content='own_prior_decomposed')
freeze_content = "own_prior"       # 'own_prior'(mean) | 'own_prior_sharp'(softmax of mean-logit) |
                                   # 'own_prior_relative'(distance-only mean) | 'own_prior_gaussian' |
                                   # 'own_prior_dirichlet' | 'random' | 'uniform' |
                                   # 'own_prior_decomposed'(fit mean) | 'random_decomposed' |
                                   # 'uniform_decomposed' (all O(T) alpha+rho storage)
freeze_pattern_dtype = "float32"   # dense fixed-pattern storage: float32 | bfloat16
                                   # bfloat16 halves the O(layers*heads*T^2) long-context buffer
resume_allow_unfrozen_pattern_dtype_change = False  # matched oracle only; rejected if any head frozen
freeze_decomp_max_kl = 0.2         # own_prior_decomposed: skip heads whose fit KL (nats/row) exceeds this
freeze_decomp_fit_steps = 400      # own_prior_decomposed: Adam steps for the batched convex fit
freeze_decomp_fit_backend = "dense"  # 'dense' reference or gated 'fft' convolution candidate
freeze_random_logit_std = 1.0      # random_decomposed: std of fixed alpha/rho logits
freeze_mode = "global"             # 'global'|'per_layer'|'threshold'|'select_once'|'significance'|
                                   # 'relative'(%-PPL budget) | 'ramp'(cheap gradual) |
                                   # 'plateau'(self-paced, train-loss-gated auto controller, no val probing)
freeze_check_interval = 500        # measure + (maybe) freeze every this many iters
freeze_warmup_frac = 0.1           # don't freeze anything before this fraction of training
freeze_max_rate = 0.5              # cap on the fraction of heads ever frozen
freeze_ramp_end = 1.0              # mode='ramp': fraction of training by which freezing reaches
                                   # max_rate (<1 leaves [ramp_end,1] for the value path to re-adapt)
freeze_p_lo = 10.0                 # variance percentile band the threshold ramps across
freeze_p_hi = 60.0
freeze_measure_batches = 16        # batches used to estimate per-head variance
freeze_measure_batch_size = 0      # calibration micro-batch; 0 uses training batch_size
                                   # (set to 1 for long-context selection memory)
freeze_variance_estimator = "sample"  # 'sample' (true per-example variance) | 'batch_mean' (legacy)
freeze_dirichlet_scale = 1.0       # multiplier on fitted row concentration (<1 noisier, >1 tighter)
freeze_dirichlet_min_concentration = 0.01
freeze_dirichlet_max_concentration = 1000000.0
freeze_val_tol = 0.0               # Marcio guard: total val-loss (nats) freezing may add
                                   # (0 = off; rate then governed by variance + max_rate)
freeze_val_rel = 0.0               # mode='relative': cumulative val-PPL increase budget,
                                   # unitless (0.01 = allow <=1% PPL); portable across sizes
freeze_val_batches = 8             # val batches per guard check
freeze_log_grad = False            # log per-head variance + Q/K grad-norm to signals.jsonl
                                   # (measure-only: study which signal predicts freezability)
freeze_select_signal = "variance"  # 'variance' | 'kl' | 'gradient' | 'combined' | 'combined_veto'
freeze_fixed_head_order = ""       # optional comma-separated L<li>H<h> order for matched arms
freeze_allow_dense_fallback = False  # diagnostic oracle only: dense prior in decomposed-capable ckpt
freeze_grad_beta = 0.9             # EMA decay for the per-head grad-norm signal
freeze_sig_t = 2.0                 # t-threshold for mode='significance' (unitless, self-calibrating)
freeze_plateau_z = 1.5             # mode='plateau': z-score for plateau/power/harm; higher = earlier
freeze_intervention_eval_batches = 0  # paired controller-val batches immediately before/after freeze
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
import pathlib as _pathlib
_this_dir = _pathlib.Path(__file__).resolve().parent
exec(open(_this_dir / 'configurator.py').read()) # overrides from command line or config file
# One-switch decomposed mode: choosing the decomposed prior REPRESENTATION
# implies the matching controller content when the content was left at its
# default (an explicitly different content still fails fast in the controller).
if (rand_attn_prior_repr == 'decomposed' and freeze_content == 'own_prior'
        and not freeze_allow_dense_fallback):
    freeze_content = 'own_prior_decomposed'
    print("rand_attn_prior_repr='decomposed' -> freeze_content auto-set to "
          "'own_prior_decomposed'")
config = {k: globals()[k] for k in config_keys} # will be useful for logging
_checkpoint_milestone_iters = {
    int(value.strip()) for value in checkpoint_milestones.split(",") if value.strip()
}
if any(step <= 0 or step > max_iters for step in _checkpoint_milestone_iters):
    raise ValueError(
        f"checkpoint_milestones must lie in [1, {max_iters}], got "
        f"{sorted(_checkpoint_milestone_iters)}")
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(base_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader — data lives at repo root
_repo_root = str(_this_dir.parent)
data_dir = os.path.join(_repo_root, 'data', dataset)
_RAM_CACHE = {}  # holds the preloaded train array when preload_train is set
_train_window_sampler = None
def get_batch(split, generator=None, consume_train_window=False, batch_count=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    suffix = '_debug' if debug_data else ''
    if split == 'train':
        if preload_train:
            if 'train' not in _RAM_CACHE:
                print("preloading train.bin into RAM ...", flush=True)
                _RAM_CACHE['train'] = np.fromfile(
                    os.path.join(data_dir, f'train{suffix}.bin'), dtype=np.uint16)
            data = _RAM_CACHE['train']
        else:
            data = np.memmap(os.path.join(data_dir, f'train{suffix}.bin'), dtype=np.uint16, mode='r')
        # Restrict sampling to the configured train window (disjoint two-stage splits).
        lo = int(train_data_start_frac * len(data))
        hi = int(train_data_end_frac * len(data)) - seq_len
        assert hi > lo, (f"empty train window: start={train_data_start_frac}, "
                         f"end={train_data_end_frac}, len={len(data)}, seq_len={seq_len}")
        sample_count = batch_size if batch_count is None else int(batch_count)
        if sample_count <= 0:
            raise ValueError("batch_count must be positive")
        if consume_train_window and train_sampling == 'shuffled_nonoverlap':
            assert _train_window_sampler is not None
            ix = _train_window_sampler.take(sample_count)
        else:
            ix = torch.randint(lo, hi, (sample_count,), generator=generator)
    else:
        data = np.memmap(os.path.join(data_dir, f'val{suffix}.bin'), dtype=np.uint16, mode='r')
        if split == 'controller':
            start_frac, end_frac = controller_val_start_frac, controller_val_end_frac
        elif split == 'val':
            start_frac, end_frac = report_val_start_frac, report_val_end_frac
        else:
            raise ValueError(f"unknown split: {split}")
        lo = int(start_frac * len(data))
        hi = int(end_frac * len(data)) - seq_len
        assert hi > lo, (f"empty {split} window: start={start_frac}, end={end_frac}, "
                         f"len={len(data)}, seq_len={seq_len}")
        sample_count = batch_size if batch_count is None else int(batch_count)
        if sample_count <= 0:
            raise ValueError("batch_count must be positive")
        ix = torch.randint(lo, hi, (sample_count,), generator=generator)
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
resume_cpu_rng_state = None
resume_cuda_rng_states = None
resume_train_rng_state = None
resume_eval_rng_state = None
resume_freeze_rng_state = None
resume_intervention_rng_state = None
resume_train_batch = None
resume_train_sampler_state = None
resume_start_iter = None

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  shared_weights=shared_weights, state_size=state_size,
                  position_encoding=position_encoding, rope_base=rope_base,
                  rand_head_rate=rand_head_rate, rand_proj_scale=rand_proj_scale,
                  rand_layer_rate=rand_layer_rate,
                  rand_layer_mode=rand_layer_mode, rand_layer_ids=rand_layer_ids,
                  rand_components=rand_components, rand_proj_mask_rate=rand_proj_mask_rate,
                  rand_attn_mask_rate=rand_attn_mask_rate,
                  rand_attn_head_rate=rand_attn_head_rate,
                  rand_attn_head_rate_per_layer=rand_attn_head_rate_per_layer,
                  rand_attn_pattern=rand_attn_pattern,
                  rand_attn_pattern_sharing=rand_attn_pattern_sharing,
                  rand_attn_window=rand_attn_window,
                  rand_attn_prior_path=rand_attn_prior_path,
                  rand_attn_dynamic=rand_attn_dynamic,
                  freeze_pattern_dtype=freeze_pattern_dtype,
                  rand_attn_prior_repr=rand_attn_prior_repr,
                  rand_seed=rand_seed)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # Handle renamed parameters from older checkpoints
    if 'rand_mask_rate' in checkpoint_model_args:
        checkpoint_model_args['rand_proj_mask_rate'] = checkpoint_model_args.pop('rand_mask_rate')
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Positional encoding is architectural, not a resume-time hyperparameter.
    # Checkpoints predating this flag are learned-absolute GPT-2 models.
    model_args['position_encoding'] = checkpoint_model_args.get(
        'position_encoding', 'learned_absolute')
    model_args['rope_base'] = checkpoint_model_args.get('rope_base', 10000.0)
    # Pattern storage dtype normally follows the checkpoint. A matched oracle
    # may change it only while every dynamic head is still unfrozen; no stored
    # pattern then exists, so this changes only the dtype of a future lazy dense
    # allocation rather than checkpoint semantics.
    checkpoint_frozen = [
        value for key, value in checkpoint['model'].items()
        if key.endswith('dyn_frozen')
    ]
    any_checkpoint_frozen = any(bool(value.any()) for value in checkpoint_frozen)
    if resume_allow_unfrozen_pattern_dtype_change:
        if any_checkpoint_frozen:
            raise ValueError(
                "cannot change freeze_pattern_dtype when resuming frozen heads")
        model_args['freeze_pattern_dtype'] = freeze_pattern_dtype
    else:
        model_args['freeze_pattern_dtype'] = checkpoint_model_args.get(
            'freeze_pattern_dtype', 'float32')
    # the frozen-prior representation decides which dyn_* buffers exist, so it
    # must match the checkpoint too (absent in pre-decomposed checkpoints)
    model_args['rand_attn_prior_repr'] = checkpoint_model_args.get(
        'rand_attn_prior_repr', 'dense')
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    packed_attention_layout = checkpoint.get('attention_layout')
    if packed_attention_layout is not None:
        if packed_attention_layout['mode'] != 'prune':
            raise ValueError('Only physical pruning uses the packed checkpoint layout')
        from supplement.attention import install_packed
        install_packed(model, packed_attention_layout['heads'], 'prune')
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    resume_start_iter = iter_num
    best_val_loss = checkpoint['best_val_loss']
    resume_cpu_rng_state = checkpoint.get('cpu_rng_state')
    resume_cuda_rng_states = checkpoint.get('cuda_rng_states')
    resume_train_rng_state = checkpoint.get('train_batch_rng_state')
    resume_eval_rng_state = checkpoint.get('eval_batch_rng_state')
    resume_freeze_rng_state = checkpoint.get('freeze_batch_rng_state')
    resume_intervention_rng_state = checkpoint.get('intervention_batch_rng_state')
    resume_train_batch = checkpoint.get('prefetched_train_batch')
    resume_train_sampler_state = checkpoint.get('train_sampler_state')
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'position_encoding', 'rope_base', 'freeze_pattern_dtype']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    # `torch.load(..., map_location=device)` leaves checkpoint tensors on the
    # accelerator. Drop the local reference to the copied model state so a
    # second 8K dynamic-pattern buffer is not retained through the selector.
    state_dict = None
checkpoint = None # free up memory

# Model construction and optimizer restoration consume RNG.  Restore the exact
# training streams only after both are complete so midpoint branches see the
# same next batch and stochastic operations as an uninterrupted run.
if resume_cpu_rng_state is not None:
    # ``torch.load(..., map_location=device)`` also moves RNG-state tensors.
    # CPU generators require a CPU ByteTensor even when the model resumes on
    # CUDA, so move every generator state back explicitly before restoring it.
    torch.random.set_rng_state(resume_cpu_rng_state.cpu())
if resume_cuda_rng_states is not None and device_type == 'cuda':
    torch.cuda.set_rng_state_all([state.cpu() for state in resume_cuda_rng_states])

# compile the model
unoptimized_model = model  # raw model: needed for the dynamic-freeze controller
if compile and rand_attn_dynamic:
    print("NOTE: rand_attn_dynamic=True -> skipping torch.compile (the controller "
          "captures attention + mutates per-head masks at runtime, which doesn't "
          "play well with compile).")
elif compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Dynamic prior buffers are immutable between explicit intervention events.
    # Broadcasting them on every forward would transfer the large derived rho
    # band repeatedly; the intervention path synchronizes and verifies them once.
    # A dynamic intervention may freeze every head in a layer. Its complete
    # Q/K projection parameters then leave the autograd graph by design. DDP
    # must discover those parameters after the intervention; otherwise the
    # next forward fails because their reduction never completed. Keeping
    # their gradients absent also prevents AdamW from changing frozen Q/K.
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        broadcast_buffers=False,
        find_unused_parameters=rand_attn_dynamic,
    )


def stateful_step(X, Y, block_size, seq_len, backward_scale=1.0):
    steps = seq_len // block_size
    loss = 0

    # DDP forwards through ``model`` but does not proxy arbitrary module
    # methods. Reset the underlying GPT state explicitly on every rank.
    raw_model.reset_state()
    for ii in range(steps):
        step_x = X[:, ii*block_size:(ii+1)*block_size]
        step_y = Y[:, ii*block_size:(ii+1)*block_size]
        s_logits, s_loss = model(step_x, step_y)
        if s_loss.requires_grad:
            scaler.scale(s_loss * (backward_scale / steps)).backward()
        loss += s_loss

    loss /= steps
    # NOTE: we deliberately do NOT assemble the full (B,T,V) logits here — they
    # were never used downstream, and building them on CPU + a GPU->CPU copy every
    # micro-step (20x/iter) was the dominant throughput cost (~3% MFU -> fixed).
    return None, loss


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, generator=eval_batch_rng)
            with ctx:
                # logits, loss = model(X, Y)
                logits, loss = stateful_step(X, Y, block_size, seq_len)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


encoder, decoder = load_tokenizer(meta_path=meta_path)
def generate_samples(prompt="\n", num_samples=1):
    model_input = encode_prompt(prompt, encoder, device)
    _ = generate(model, model_input, ctx, decoder, num_samples=num_samples)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_rng = torch.Generator().manual_seed(base_seed + seed_offset + 10_003)
eval_batch_rng = torch.Generator().manual_seed(base_seed + seed_offset + 20_003)
if resume_train_rng_state is not None:
    train_batch_rng.set_state(resume_train_rng_state.cpu())
if resume_eval_rng_state is not None:
    eval_batch_rng.set_state(resume_eval_rng_state.cpu())
if train_sampling not in ('random_with_replacement', 'shuffled_nonoverlap'):
    raise ValueError(f"unknown train_sampling={train_sampling!r}")
if train_sampling == 'shuffled_nonoverlap':
    train_path = os.path.join(
        data_dir, f"train{'_debug' if debug_data else ''}.bin")
    train_tokens = os.path.getsize(train_path) // np.dtype(np.uint16).itemsize
    sampler_lo = int(train_data_start_frac * train_tokens)
    sampler_end = int(train_data_end_frac * train_tokens)
    _train_window_sampler = ShuffledNonoverlapSampler(
        lo=sampler_lo, end=sampler_end, window_size=seq_len,
        seed=base_seed + 30_003,
        rank=(ddp_rank if ddp else 0), world_size=ddp_world_size)
    if resume_train_sampler_state is not None:
        _train_window_sampler.load_state_dict(resume_train_sampler_state)
    # The loop always keeps one micro-batch prefetched in the exact-resume
    # checkpoint, including after the final optimizer update. That last batch
    # is reserved but not counted as a processed training example.
    required_windows = (
        max_iters * gradient_accumulation_steps * batch_size + batch_size)
    if required_windows > _train_window_sampler.rank_windows:
        raise ValueError(
            f"rank {(ddp_rank if ddp else 0)} needs {required_windows:,} disjoint "
            f"windows but its partition contains only "
            f"{_train_window_sampler.rank_windows:,}")
    if master_process:
        print(
            "training sampler: shuffled_nonoverlap "
            f"global_windows={_train_window_sampler.n_windows:,} "
            f"world_size={ddp_world_size} "
            f"global_capacity_tokens={_train_window_sampler.n_windows * seq_len:,}",
            flush=True)
if resume_train_batch is None:
    X, Y = get_batch('train', generator=train_batch_rng, consume_train_window=True)
else:
    X, Y = (tensor.to(device) for tensor in resume_train_batch)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
pbar = tqdm(total=max_iters, initial=iter_num, desc="training", disable=not master_process)

import json as _json
# Local metrics JSONL — written alongside wandb so we can scp directly
_metrics_path = os.path.join(out_dir, 'metrics.jsonl') if master_process else None
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # Truncate / start fresh
    open(_metrics_path, 'w').close()
def _log_local(d):
    if _metrics_path is None: return
    with open(_metrics_path, 'a') as _f:
        _f.write(_json.dumps(d) + '\n')

# Compute static FLOPs estimate once (same for every iteration)
_flops_actual, _flops_baseline = raw_model.flops_per_iter(
    fwdbwd_per_iter=batch_size * gradient_accumulation_steps)
_flops_savings_pct = 100 * (1 - _flops_actual / _flops_baseline)
print(f"FLOPs/iter: actual={_flops_actual/1e9:.2f} GFLOPs, "
      f"baseline={_flops_baseline/1e9:.2f} GFLOPs ({_flops_savings_pct:+.1f}% vs baseline)")
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

# dynamic per-head freezing controller (Version A: gradual attention training)
freeze_ctrl = None
freeze_batch_rng = torch.Generator().manual_seed(base_seed + seed_offset + 100_003)
if resume_freeze_rng_state is not None:
    freeze_batch_rng.set_state(resume_freeze_rng_state.cpu())
intervention_batch_rng = torch.Generator().manual_seed(
    base_seed + seed_offset + 100_019)
if resume_intervention_rng_state is not None:
    intervention_batch_rng.set_state(resume_intervention_rng_state.cpu())
def get_freeze_batch(split):
    # Controller validation never consumes the untouched reporting partition.
    mapped = 'controller' if split == 'val' else split
    calibration_batch = (
        freeze_measure_batch_size if freeze_measure_batch_size > 0 else batch_size)
    return get_batch(
        mapped, generator=freeze_batch_rng, batch_count=calibration_batch)
def get_intervention_batch():
    # Paired before/after loss probing must not perturb the selector's RNG.
    # In particular, matched continuations must capture exactly the same
    # measurement windows as the reusable fitter-statistics run.
    calibration_batch = (
        freeze_measure_batch_size if freeze_measure_batch_size > 0 else batch_size)
    return get_batch(
        'controller', generator=intervention_batch_rng,
        batch_count=calibration_batch)
freeze_control_group = None
if ddp and rand_attn_dynamic and packed_attention_layout is None:
    if (freeze_mode != "select_once" or rand_attn_prior_repr != "decomposed"
            or freeze_allow_dense_fallback):
        raise ValueError(
            "DDP dynamic freezing is deliberately restricted to select_once with "
            "a compact decomposed prior and no dense fallback")
    if freeze_max_rate > 0:
        from distributed_freeze import create_freeze_control_group
        freeze_control_group = create_freeze_control_group()
if rand_attn_dynamic and packed_attention_layout is None:
    from dynamic_freeze import DynamicFreezeController
    freeze_ctrl = DynamicFreezeController(
        raw_model, content=freeze_content, mode=freeze_mode,
        check_interval=freeze_check_interval,
        warmup_frac=freeze_warmup_frac, max_rate=freeze_max_rate, ramp_end=freeze_ramp_end,
        p_lo=freeze_p_lo, p_hi=freeze_p_hi, measure_batches=freeze_measure_batches,
        val_tol=freeze_val_tol, val_rel=freeze_val_rel, val_batches=freeze_val_batches,
        select_signal=freeze_select_signal, grad_beta=freeze_grad_beta,
        sig_t=freeze_sig_t, plateau_z=freeze_plateau_z,
        variance_estimator=freeze_variance_estimator,
        dirichlet_scale=freeze_dirichlet_scale,
        dirichlet_min_concentration=freeze_dirichlet_min_concentration,
        dirichlet_max_concentration=freeze_dirichlet_max_concentration,
        decomp_max_kl=freeze_decomp_max_kl, decomp_fit_steps=freeze_decomp_fit_steps,
        decomp_fit_backend=freeze_decomp_fit_backend,
        random_logit_std=freeze_random_logit_std,
        fixed_head_order=freeze_fixed_head_order,
        allow_dense_fallback=freeze_allow_dense_fallback,
        seed=rand_seed,
        log=(None if master_process else (lambda *args, **kwargs: None)))
    if master_process:
        print(f"dynamic freeze: mode={freeze_mode} content={freeze_content} "
              f"every {freeze_check_interval} iters, max_rate={freeze_max_rate}, "
              f"warmup_frac={freeze_warmup_frac}, val_tol={freeze_val_tol}, "
              f"val_rel={freeze_val_rel}, variance_estimator={freeze_variance_estimator}, "
              f"select_signal={freeze_select_signal}, "
              f"measure_batch_size={freeze_measure_batch_size or batch_size}",
              flush=True)

freeze_profile_events = []
_run_started_at = time.time()
_milestone_dir = os.path.join(out_dir, "checkpoints")
if master_process and _checkpoint_milestone_iters:
    os.makedirs(_milestone_dir, exist_ok=True)


def _checkpoint_payload():
    """Build one exact-resume checkpoint payload.

    In particular, preserve the already-prefetched batch and every independent
    RNG stream. Paper experiments branch several continuations from the same
    baseline milestone; omitting any of these fields makes those branches only
    approximately, rather than exactly, paired.
    """
    return {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'attention_layout': packed_attention_layout,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'cpu_rng_state': torch.random.get_rng_state(),
        'cuda_rng_states': (
            torch.cuda.get_rng_state_all() if device_type == 'cuda' else None
        ),
        'train_batch_rng_state': train_batch_rng.get_state(),
        'eval_batch_rng_state': eval_batch_rng.get_state(),
        'freeze_batch_rng_state': freeze_batch_rng.get_state(),
        'intervention_batch_rng_state': intervention_batch_rng.get_state(),
        # A DDP checkpoint is written only by rank 0. For shuffled sampling,
        # store the shared cursor immediately before the prefetched batch so
        # every rank reconstructs its own exact next batch on resume.
        'prefetched_train_batch': (
            None if ddp and train_sampling == 'shuffled_nonoverlap'
            else (X.detach().cpu(), Y.detach().cpu())
        ),
        'train_sampler_state': (
            _train_window_sampler.state_dict(
                cursor_rewind=(
                    batch_size
                    if ddp and train_sampling == 'shuffled_nonoverlap' else 0))
            if _train_window_sampler is not None else None),
        'experiment': {
            'run_id': os.environ.get('EXPERIMENT_RUN_ID'),
            'manifest_sha256': os.environ.get('EXPERIMENT_MANIFEST_SHA256'),
            'source_state_sha256': os.environ.get('SOURCE_STATE_SHA256'),
        },
    }


def _save_checkpoint(path):
    tmp = path + ".tmp"
    torch.save(_checkpoint_payload(), tmp)
    os.replace(tmp, path)
    print(f"saved exact checkpoint: {path}", flush=True)

while True:

    # Full baselines retain exact branch points for the rate/timing/scheduler
    # matrix. Save before any intervention at this iteration. Formal manifests
    # only request milestones on unfrozen baselines, but keeping the ordering
    # explicit makes the contract unambiguous.
    if (master_process and iter_num in _checkpoint_milestone_iters
            and iter_num != resume_start_iter):
        milestone_path = os.path.join(_milestone_dir, f"iter_{iter_num:07d}.pt")
        if not os.path.exists(milestone_path):
            _save_checkpoint(milestone_path)

    # dynamic freeze check (measures variance + may freeze stable heads)
    if freeze_ctrl is not None:
        # The one-shot selector materializes attention matrices and has a very
        # different memory profile from steady training. Profile it separately,
        # then reset CUDA peak stats so frozen-arm VRAM remains comparable.
        selector_due = (
            freeze_ctrl.mode == 'select_once'
            and freeze_ctrl.max_rate > 0
            and iter_num >= int(freeze_ctrl.warmup_frac * max_iters)
            and iter_num % freeze_ctrl.M == 0
            and freeze_ctrl.current_rate() == 0
        )
        profile_selector = device_type == 'cuda' and master_process and selector_due
        intervention_batches = None
        intervention_pre_nll = None
        intervention_pre_eval_s = 0.0
        if profile_selector and freeze_intervention_eval_batches > 0:
            if device_type == 'cuda':
                torch.cuda.synchronize()
            intervention_pre_t0 = time.time()
            intervention_batches = [
                get_intervention_batch()
                for _ in range(freeze_intervention_eval_batches)
            ]
            intervention_pre_nll = freeze_ctrl._val_loss_fixed(
                intervention_batches, ctx)
            if device_type == 'cuda':
                torch.cuda.synchronize()
            intervention_pre_eval_s = time.time() - intervention_pre_t0
        if profile_selector:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            selector_t0 = time.time()
        freeze_state_sha256 = None
        if master_process:
            freeze_ctrl.maybe_freeze(iter_num, max_iters, get_freeze_batch, ctx)
        if ddp and selector_due:
            from distributed_freeze import synchronize_decomposed_freeze, wait_for_freeze_fit
            # Calibration can take longer than NCCL's watchdog allows an
            # early broadcast to wait. Synchronize on the host first.
            wait_for_freeze_fit(freeze_control_group)
            if master_process:
                print("[freeze] calibration complete; broadcasting compact state", flush=True)
            freeze_state_sha256 = synchronize_decomposed_freeze(freeze_ctrl.mods)
        if profile_selector:
            torch.cuda.synchronize()
            selector_total_s = float(time.time() - selector_t0)
            selector_peak_vram_mb = float(
                torch.cuda.max_memory_allocated() / 1e6)
            torch.cuda.reset_peak_memory_stats()
            intervention_post_nll = None
            intervention_post_eval_s = 0.0
            if intervention_batches is not None:
                intervention_post_t0 = time.time()
                intervention_post_nll = freeze_ctrl._val_loss_fixed(
                    intervention_batches, ctx)
                torch.cuda.synchronize()
                intervention_post_eval_s = time.time() - intervention_post_t0
            selector_phases = {
                key: float(value)
                for key, value in freeze_ctrl.last_selector_phases.items()
            }
            top_level_accounted = sum(
                selector_phases.get(key, 0.0)
                for key in ("capture_and_mean_s", "ranking_s", "freeze_batch_total_s")
            )
            selector_event = {
                "event": "freeze_selector",
                "iter": int(iter_num),
                "freeze/selector_time_s": selector_total_s,
                "freeze/selector_peak_vram_mb": selector_peak_vram_mb,
                "freeze/capture_dtype": freeze_ctrl.last_capture_dtype,
                "freeze/frozen_rate_after": float(freeze_ctrl.current_rate()),
                "freeze/ddp_state_sha256": freeze_state_sha256,
                "freeze/selector_phases_s": selector_phases,
                "freeze/selector_unattributed_s": max(
                    0.0, selector_total_s - top_level_accounted),
            }
            if intervention_pre_nll is not None and intervention_post_nll is not None:
                delta_nll = intervention_post_nll - intervention_pre_nll
                selector_event.update({
                    "freeze/intervention_eval_batches": int(
                        freeze_intervention_eval_batches),
                    "freeze/intervention_pre_nll": float(intervention_pre_nll),
                    "freeze/intervention_post_nll": float(intervention_post_nll),
                    "freeze/intervention_delta_nll": float(delta_nll),
                    "freeze/intervention_delta_ppl_pct": float(
                        math.expm1(delta_nll) * 100.0),
                    "freeze/intervention_eval_time_s": float(
                        intervention_pre_eval_s + intervention_post_eval_s),
                })
            if freeze_ctrl.last_decomp_stats_chunks:
                selector_stats_path = os.path.join(out_dir, "selector_stats.pt")
                selector_stats_tmp = selector_stats_path + ".tmp"
                torch.save({
                    "schema_version": 1,
                    "iter": int(iter_num),
                    "content": freeze_content,
                    "selection_signal": freeze_select_signal,
                    "variance_estimator": freeze_variance_estimator,
                    "context": int(seq_len),
                    "decomp_fit_steps": int(freeze_decomp_fit_steps),
                    "decomp_fit_backend": freeze_decomp_fit_backend,
                    "decomp_max_kl": float(freeze_decomp_max_kl),
                    "head_variance": freeze_ctrl.last_var,
                    "chunks": freeze_ctrl.last_decomp_stats_chunks,
                }, selector_stats_tmp)
                os.replace(selector_stats_tmp, selector_stats_path)
                selector_event["freeze/selector_stats_file"] = "selector_stats.pt"
                selector_event["freeze/selector_stats_bytes"] = int(
                    os.path.getsize(selector_stats_path))
            freeze_profile_events.append(selector_event)
            _log_local(selector_event)
            print(f"selector profile: {selector_event}", flush=True)
            torch.cuda.reset_peak_memory_stats()

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    skip_resumed_eval = (
        init_from == 'resume' and skip_initial_eval_on_resume
        and iter_num == resume_start_iter
    )
    if ((iter_num % eval_interval == 0 or iter_num >= max_iters)
            and master_process and not skip_resumed_eval):
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        generate_samples()
        eval_log = {
            "iter": iter_num,
            "train/loss": float(losses['train']),
            "val/loss": float(losses['val']),
            "lr": float(lr),
            "mfu": float(running_mfu * 100),
        }
        _log_local(eval_log)
        if wandb_log:
            wandb.log(eval_log)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                _save_checkpoint(os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    # ``max_iters`` is the number of optimizer updates.  Evaluation happens at
    # the start of an iteration, so save/evaluate the state after the final
    # update and stop before accidentally applying update max_iters + 1.
    if iter_num >= max_iters:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # logits, loss = model(X, Y)
            backward_scale = (1.0 / gradient_accumulation_steps
                              if normalize_gradient_accumulation else 1.0)
            logits, loss = stateful_step(
                X, Y, block_size, seq_len, backward_scale=backward_scale)
            # loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch(
            'train', generator=train_batch_rng, consume_train_window=True)
        # backward pass, with gradient scaling if training in fp16
        # this was moved to stateful_step for now
        # scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # update the per-head grad-norm EMA every step (convergence signal for selection)
    needs_grad_signal = freeze_select_signal in ("gradient", "combined", "combined_veto")
    if freeze_ctrl is not None and (needs_grad_signal or freeze_log_grad):
        freeze_ctrl.update_grad_ema()
    # mode='plateau': feed the free per-step train loss to the auto controller (self-paced)
    if freeze_ctrl is not None and freeze_mode == 'plateau':
        freeze_ctrl.observe_loss(iter_num, max_iters, float(loss), get_freeze_batch, ctx)
    # measure-only: log per-head variance + Q/K grad-norm signals (grads ready here)
    if (freeze_ctrl is not None and freeze_log_grad and master_process
            and iter_num % freeze_check_interval == 0):
        _g = freeze_ctrl.per_head_grad_norm()
        if _g is not None:
            _v = freeze_ctrl.last_var
            _log_local({"iter": iter_num, "signals/grad_norm": _g.tolist(),
                        "signals/variance": (_v.tolist() if _v is not None else None)})
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item()
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device_type == 'cuda' else 0.0
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved() / 1e6 if device_type == 'cuda' else 0.0)
        pbar.set_postfix(loss=f"{lossf:.4f}", lr=f"{lr:.1e}",
                         mfu=f"{running_mfu*100:.2f}%",
                         vram=f"{peak_vram_mb:.0f}MB",
                         t=f"{dt*1000:.0f}ms")
        log_dict = {
            "iter": iter_num,
            "train/loss_step": float(lossf),
            "lr": float(lr),
            "perf/iter_time_ms": float(dt * 1000),
            "perf/peak_vram_mb": float(peak_vram_mb),
            "perf/peak_vram_reserved_mb": float(peak_reserved_mb),
            "perf/flops_per_iter_G": float(_flops_actual / 1e9),
            "perf/flops_savings_pct": float(_flops_savings_pct),
        }
        if running_mfu > 0:
            log_dict["perf/mfu"] = float(running_mfu * 100)
        _log_local(log_dict)
        if wandb_log:
            wandb.log(log_dict)
    pbar.update(1)
    iter_num += 1
    local_iter_num += 1

pbar.close()

# Final summary — easy to scp and parse
if master_process:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = None
    summary = {
        "out_dir": out_dir,
        "dataset": dataset,
        "best_val_loss": float(best_val_loss),
        "final_iter": int(iter_num),
        "flops_per_iter_G": float(_flops_actual / 1e9),
        "flops_baseline_G": float(_flops_baseline / 1e9),
        "flops_savings_pct": float(_flops_savings_pct),
        "wandb_run_name": wandb_run_name if 'wandb_run_name' in globals() else None,
        "model_args": {k: (v if not callable(v) else str(v)) for k, v in model_args.items()},
        "model_parameters": int(raw_model.get_num_params(non_embedding=False)),
        "trainable_parameters": int(sum(
            parameter.numel() for parameter in raw_model.parameters()
            if parameter.requires_grad)),
        "attention_layout": packed_attention_layout,
        "flops_estimate_includes_physical_pruning": False,
        "config": config,
        "git_sha": git_sha,
        "source_commit": os.environ.get("SOURCE_COMMIT"),
        "source_state_sha256": os.environ.get("SOURCE_STATE_SHA256"),
        "experiment_run_id": os.environ.get("EXPERIMENT_RUN_ID"),
        "experiment_manifest_sha256": os.environ.get("EXPERIMENT_MANIFEST_SHA256"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_time_s": float(time.time() - _run_started_at),
        "checkpoint_milestones": sorted(_checkpoint_milestone_iters),
        "checkpoint_files": [
            str(pathlib.Path("checkpoints") / f"iter_{step:07d}.pt")
            for step in sorted(_checkpoint_milestone_iters)
            if os.path.exists(os.path.join(_milestone_dir, f"iter_{step:07d}.pt"))
        ],
        "final_checkpoint_bytes": (
            os.path.getsize(os.path.join(out_dir, "ckpt.pt"))
            if os.path.exists(os.path.join(out_dir, "ckpt.pt")) else None
        ),
        "train_sampling": train_sampling,
        "training_tokens_processed": int(iter_num * tokens_per_iter),
        "training_unique_target_tokens_by_construction": (
            int(iter_num * tokens_per_iter)
            if train_sampling == 'shuffled_nonoverlap' else None),
        "train_sampler": (
            _train_window_sampler.state_dict()
            if _train_window_sampler is not None else None),
    }
    if freeze_ctrl is not None:
        frozen_per_layer = [int(m.dyn_frozen.sum()) for m in freeze_ctrl.mods]
        summary["freeze"] = {
            "content": freeze_content,
            "mode": freeze_mode,
            "prior_repr": rand_attn_prior_repr,
            "effective_prior_repr": (
                "dense" if freeze_allow_dense_fallback else rand_attn_prior_repr),
            "variance_estimator": freeze_variance_estimator,
            "select_signal": freeze_select_signal,
            "measure_batches": int(freeze_measure_batches),
            "measure_batch_size": int(
                freeze_measure_batch_size or batch_size),
            "frozen_per_layer": frozen_per_layer,
            "frozen_rate": sum(frozen_per_layer) / max(freeze_ctrl.total_heads, 1),
            "frozen_heads": [
                f"L{li}H{h}"
                for li, module in enumerate(freeze_ctrl.mods)
                for h in range(freeze_ctrl.n_head)
                if bool(module.dyn_frozen[h])
            ],
            "last_head_variance": (
                freeze_ctrl.last_var.tolist() if freeze_ctrl.last_var is not None else None
            ),
            "last_head_forward_kl": (
                freeze_ctrl.last_kl.tolist() if freeze_ctrl.last_kl is not None else None
            ),
            "selector_profile": freeze_profile_events,
            "fixed_head_order": [
                f"L{idx // freeze_ctrl.n_head}H{idx % freeze_ctrl.n_head}"
                for idx in freeze_ctrl.fixed_head_order
            ],
            "decomposed_fit_kl": {
                f"L{li}H{h}": float(value)
                for (li, h), value in sorted(freeze_ctrl.last_decomp_kl.items())
            },
            "decomposed_skipped_count": int(freeze_ctrl.decomp_skipped),
            "decomposed_fit_backend": freeze_ctrl.decomp_fit_backend,
        }
        prior_buffer_names = {
            "dyn_pattern", "dyn_alpha", "dyn_rho", "dyn_z", "dyn_rho_band"
        }
        prior_bytes = 0
        for name, value in raw_model.named_buffers():
            if name.rsplit(".", 1)[-1] in prior_buffer_names and value is not None:
                prior_bytes += value.numel() * value.element_size()
        summary["freeze"]["stored_prior_bytes"] = int(prior_bytes)
        concentration = freeze_ctrl.last_dirichlet_concentration
        if concentration is not None:
            finite_concentration = torch.isfinite(concentration)
            values = concentration[finite_concentration]
            fallback = freeze_ctrl.last_dirichlet_fallback
            summary["freeze"]["dirichlet_concentration_p10_p50_p90"] = [
                float(torch.quantile(values, q)) for q in (0.1, 0.5, 0.9)
            ] if values.numel() else []
            summary["freeze"]["dirichlet_deterministic_row_fraction"] = \
                float(fallback[finite_concentration].float().mean()) \
                if fallback is not None and values.numel() else None
    if device_type == 'cuda':
        summary["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
        summary["peak_vram_reserved_mb"] = float(
            torch.cuda.max_memory_reserved() / 1e6)
    summary["fused_dispatch_count_rank0"] = int(fused_dispatch_count())
    with open(os.path.join(out_dir, 'summary.json'), 'w') as _f:
        _json.dump(summary, _f, indent=2)
    print(f"Wrote summary to {out_dir}/summary.json")

if ddp:
    destroy_process_group()

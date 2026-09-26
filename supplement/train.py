"""Run one pinned supplementary arm with explicit cost and checkpoint records."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanogpt.model import GPT, GPTConfig, fused_dispatch_count
from nanogpt.data_sampling import ShuffledNonoverlapSampler
from nanogpt.dynamic_freeze import DynamicFreezeController
from supplement.attention import install_packed, random_eligible_heads
from supplement.protocol import clock_lr, token_lr, continuation_budget


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


class Windows:
    def __init__(self, data_dir, cfg, seed, device):
        self.cfg, self.device = cfg, device
        self.train = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
        self.val = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
        self.sampler = ShuffledNonoverlapSampler(
            lo=0, end=len(self.train), window_size=cfg['context'], seed=seed + 30003)
        self.calibration_rng = torch.Generator().manual_seed(seed + 100003)
        self.half = len(self.val) // 2
        self.report_starts = list(range(self.half, len(self.val) - cfg['context'], cfg['context']))
        if cfg.get('pilot'):
            self.report_starts = self.report_starts[:8]
        if not self.report_starts:
            raise ValueError('reporting interval is too short')

    def batch(self, data, starts):
        length = self.cfg['context']
        x = torch.from_numpy(np.stack([data[int(i):int(i)+length] for i in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[int(i)+1:int(i)+1+length] for i in starts]).astype(np.int64))
        if self.device == 'cuda':
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x, y

    def training(self):
        return self.batch(self.train, self.sampler.take(self.cfg['batch']))

    def calibration(self, split):
        starts = torch.randint(self.half - self.cfg['context'],
                               (self.cfg['calibration_batch'],), generator=self.calibration_rng)
        return self.batch(self.val, starts)


@torch.no_grad()
def evaluate(model, windows, context, complete=False):
    was_training = model.training
    model.eval()
    starts = windows.report_starts
    if not complete:
        starts = starts[:windows.cfg['monitor_windows']]
    nll, tokens = 0., 0
    # Identical windows at every checkpoint; no sampling or padding.
    for offset in range(0, len(starts), windows.cfg['calibration_batch']):
        x, y = windows.batch(windows.val, starts[offset:offset+windows.cfg['calibration_batch']])
        with context():
            _, loss = model(x, y)
        nll += float(loss) * y.numel()
        tokens += y.numel()
    model.train(was_training)
    return {'nll': nll / tokens, 'ppl': math.exp(nll / tokens), 'tokens': tokens}


def intervene(model, optimizer, windows, context, arm, seed, cfg):
    if arm['mode'] == 'ordinary':
        return {'mode': 'ordinary', 'heads': [[] for _ in model.transformer.h], 'total_s': 0.}
    device = windows.device
    started = time.perf_counter()
    controller = DynamicFreezeController(
        model, content='own_prior_decomposed', mode='select_once',
        measure_batches=cfg['calibration_batches'], variance_estimator='sample',
        decomp_fit_steps=cfg['fit_steps'], decomp_fit_backend='fft',
        decomp_max_kl=cfg['fit_max_kl'])
    scores, means = controller._measure(windows.calibration, context())
    sync(device)
    captured = time.perf_counter()
    count = round(cfg['n_layer'] * cfg['n_head'] * arm['rate'])
    ranking = sorted(range(scores.numel()), key=lambda i: (float(scores.flatten()[i]), i))
    reference = [[i % cfg['n_head'] for i in ranking[:count] if i // cfg['n_head'] == layer]
                 for layer in range(cfg['n_layer'])]
    reference = [sorted(heads) for heads in reference]
    heads = reference
    ranked = time.perf_counter()
    if arm['mode'] in ('mean', 'random_mean'):
        controller._profile_selector_phases = True
        causal = torch.tril(torch.ones(cfg['context'], cfg['context']))
        if arm['mode']=='random_mean':
            heads=random_eligible_heads(reference,cfg['n_head'],seed+910003,
                lambda pairs:controller._freeze_batch(pairs,means,cfg['context'],causal))
        else:
            pairs = [(li, h) for li, chosen in enumerate(heads) for h in chosen]
            installed = controller._freeze_batch(pairs, means, cfg['context'], causal)
            if set(installed) != set(pairs):
                raise RuntimeError('a fixed variance head failed the fit gate; no silent substitution')
    else:
        install_packed(model, heads, arm['mode'], optimizer)
    sync(device)
    fitted = time.perf_counter()
    result = dict(mode=arm['mode'], heads=heads, variance_reference_heads=reference,
                  per_layer_counts=[len(h) for h in heads], head_variance=scores.tolist(),
                  capture_s=captured-started, ranking_s=ranked-captured,
                  fitting_installation_s=fitted-ranked, total_s=fitted-started,
                  fit_phases_s=controller.last_selector_phases,
                  fit_kl={f'L{li}H{h}': value for (li,h),value in controller.last_decomp_kl.items()},
                  random_head_seed=seed+910003 if arm['mode']=='random_mean' else None,
                  fit_policy=('random candidate order within layer; common fit gate and matched accepted counts' if arm['mode']=='random_mean'
                              else 'exact variance set; abort if any fit exceeds threshold'))
    del means, controller
    gc.collect()
    return result


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--prefix-receipt', type=Path)
    parser.add_argument('--ordinary-summary', type=Path)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--learn-patterns', action='store_true',
                        help='Continue learning fitted alpha/rho for a mean arm')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cfg = manifest['settings']
    arm = next(a for a in manifest['arms'] if a['name'] == args.arm)
    if args.out.exists():
        raise RuntimeError(f'refusing to overwrite an existing run: {args.out}')
    args.out.mkdir(parents=True)
    write_json(args.out/'arm.json', dict(arm=arm, settings=cfg, seed=args.seed,
                                      manifest_sha256=sha256(args.manifest),
                                      slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    context = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if device=='cuda' else nullcontext
    model_cfg = GPTConfig(block_size=cfg['context'], n_layer=cfg['n_layer'],
                          n_head=cfg['n_head'], n_embd=cfg['n_embd'], vocab_size=cfg['vocab_size'],
                          dropout=0., bias=False, position_encoding='rope',
                          rand_attn_dynamic=True, rand_attn_prior_repr='decomposed',
                          freeze_pattern_dtype='bfloat16')
    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(cfg['weight_decay'], cfg['peak_lr'],
                                          (cfg['beta1'], cfg['beta2']), device)
    windows = Windows(args.data, cfg, args.seed, device)
    step, prefix_cost, parent_hash = 0, 0., None
    if args.parent:
        parent_hash = sha256(args.parent)
        receipt = json.loads(args.prefix_receipt.read_text())
        if receipt['sha256'] != parent_hash:
            raise RuntimeError('parent checkpoint hash mismatch')
        state = torch.load(args.parent, map_location=device, weights_only=False)
        if state['intervention']['mode'] != 'ordinary':
            raise RuntimeError('all branches must start from an ordinary checkpoint')
        if state['model_config'] != asdict(model_cfg):
            raise RuntimeError('model configuration differs from the parent')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        windows.sampler.load_state_dict(state['sampler'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda': torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        step = state['step']
        prefix_cost = receipt['elapsed_s']
        if windows.sampler.cursor != step * cfg['batch'] * cfg['accumulation']:
            raise RuntimeError('parent sampler cursor does not match processed updates')
        del state
    start_step = step
    budget_s = None
    if arm['budget'] == 'clock':
        ordinary = json.loads(args.ordinary_summary.read_text())
        if ordinary['seed'] != args.seed or ordinary['final_step'] != cfg['updates']:
            raise RuntimeError('invalid ordinary budget reference')
        budget_s = continuation_budget(ordinary['training_elapsed_s'], prefix_cost)
        if cfg.get('pilot'):
            budget_s = max(budget_s, cfg.get('pilot_clock_budget_s', 180.))
    loaded_s = time.perf_counter() - process_started
    intervention = intervene(model, optimizer, windows, context, arm, args.seed, cfg)
    if args.learn_patterns:
        if arm['mode'] != 'mean':
            raise ValueError('--learn-patterns requires a mean arm')
        from supplement.trainable_patterns import enable
        enable(model, optimizer)
        intervention['trainable_patterns'] = True
    write_json(args.out/'intervention.json', intervention)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    def elapsed():
        return time.perf_counter() - process_started
    metrics = (args.out/'metrics.jsonl').open('w', buffering=1)
    def log(event):
        metrics.write(json.dumps(event, allow_nan=False) + '\n')
    checkpoints = {}
    def save(path, kind):
        sync(device)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       model_config=asdict(model_cfg), settings=cfg, step=step,
                       sampler=windows.sampler.state_dict(), intervention=intervention,
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None)
        temporary = path.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        digest = sha256(path)
        receipt = dict(path=str(path), sha256=digest, step=step,
                       elapsed_s=prefix_cost+elapsed(), kind=kind)
        write_json(path.with_suffix('.receipt.json'), receipt)
        checkpoints[kind] = receipt
    def monitor(kind):
        before = elapsed()
        score = evaluate(model, windows, context)
        log(dict(event='evaluation', kind=kind, step=step, prefix_s=prefix_cost,
                 elapsed_s=elapsed(), total_elapsed_s=prefix_cost+elapsed(),
                 model_elapsed_s=prefix_cost+before,
                 evaluation_s=elapsed()-before, **score))
    model.train()
    monitor('initial')
    next_clock_eval = .1
    last_update_s = 0.
    update_times = []
    cumulative_update_s, cumulative_loading_s = 0., 0.
    last_training_elapsed = elapsed()
    stop_reason = 'updates'
    # The same backend restriction applies to every ordinary head in every arm.
    backend_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device=='cuda' else nullcontext()
    with backend_context:
        while True:
            if arm['budget']=='tokens' and step >= cfg['updates']:
                break
            if budget_s is not None and elapsed() + max(1., last_update_s * 1.5) >= budget_s:
                stop_reason = 'wall_clock'
                break
            if step >= (1000 if cfg.get('pilot') else cfg['updates'] * 3):
                raise RuntimeError('clock run exceeded the declared token safety cap')
            lr = (clock_lr(start_step, elapsed(), budget_s, cfg) if budget_s is not None
                  else token_lr(step, cfg))
            for group in optimizer.param_groups: group['lr'] = lr
            sync(device)
            update_started = time.perf_counter()
            losses, loading_s = [], 0.
            for _ in range(cfg['accumulation']):
                load_started = time.perf_counter()
                x, y = windows.training()
                loading_s += time.perf_counter() - load_started
                with context():
                    _, loss = model(x, y)
                    scaled = loss / cfg['accumulation']
                scaled.backward()
                losses.append(loss.detach())
            loss_total = float(torch.stack(losses).mean())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            if not math.isfinite(loss_total) or not math.isfinite(float(norm)):
                raise RuntimeError('non-finite loss or gradient norm')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            last_update_s = time.perf_counter() - update_started
            cumulative_update_s += last_update_s
            cumulative_loading_s += loading_s
            update_times.append(last_update_s)
            step += 1
            last_training_elapsed = elapsed()
            if budget_s is not None and last_training_elapsed > budget_s:
                raise RuntimeError('atomic update exceeded wall-clock budget; result is invalid')
            log(dict(event='update', step=step, loss=loss_total, lr=lr,
                     gradient_norm=float(norm), update_s=last_update_s, loading_s=loading_s,
                     elapsed_s=last_training_elapsed,
                     total_elapsed_s=prefix_cost+last_training_elapsed,
                     training_tokens=step*cfg['batch']*cfg['accumulation']*cfg['context']))
            if step % max(1,cfg['eval_every']) == 0:
                print(f'{args.arm} seed={args.seed} step={step} loss={loss_total:.5f} update_s={last_update_s:.3f}', flush=True)
            if arm['parent'] is None and step in (cfg['updates']//4, cfg['updates']//2):
                kind = 'quarter' if step==cfg['updates']//4 else 'midpoint'
                save(args.out/f'{kind}.pt', kind)
            if budget_s is None:
                if step % cfg['eval_every']==0 and step < cfg['updates']:
                    monitor('trajectory')
            elif elapsed()/budget_s >= next_clock_eval:
                # Leave the last slice for updates; reporting evaluation is outside
                # the training budget and timed explicitly below for every method.
                if elapsed() + max(15., last_update_s * 10) < budget_s:
                    monitor('trajectory')
                next_clock_eval = (math.floor(elapsed()/budget_s*10)+1)/10
    sync(device)
    training_end_s = elapsed()
    if not update_times:
        raise RuntimeError('budget left no room for a continuation update')
    if intervention['mode'] in ('mean','random_mean') and device=='cuda' and fused_dispatch_count()==0:
        raise RuntimeError('compact mean did not use the fused kernel')
    monitor('endpoint_monitor')
    final_eval_start = elapsed()
    final_score = evaluate(model, windows, context, complete=True)
    final_eval_s = elapsed()-final_eval_start
    save(args.out/'final.pt', 'final')
    summary = dict(seed=args.seed, arm=arm, settings=cfg, final_step=step,
                   parent_step=start_step, parent_sha256=parent_hash, prefix_s=prefix_cost,
                   continuation_budget_s=budget_s, training_elapsed_s=prefix_cost+training_end_s,
                   last_update_elapsed_s=prefix_cost+last_training_elapsed,
                   continuation_training_s=training_end_s, process_total_s=elapsed(),
                   setup_loading_s=loaded_s, intervention=intervention,
                   updates_s=cumulative_update_s, data_loading_in_updates_s=cumulative_loading_s,
                   last_100_mean_update_s=float(np.mean(update_times[-100:])),
                   final_reporting_evaluation_s=final_eval_s, final_score=final_score,
                   final_checkpoint_sha256=checkpoints['final']['sha256'],
                   model_parameters=sum(p.numel() for p in model.parameters()),
                   optimiser_state_elements=sum(v.numel() for s in optimizer.state.values()
                                                for v in s.values() if isinstance(v,torch.Tensor)),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                   fused_dispatches=fused_dispatch_count(), stop_reason=stop_reason,
                   training_tokens=step*cfg['batch']*cfg['accumulation']*cfg['context'],
                   sampler=windows.sampler.state_dict(), checkpoints=checkpoints,
                   timing_definition='Includes setup, data loading, calibration, fitting, installation, monitoring and intermediate checkpointing; final reporting evaluation/export are separate.')
    write_json(args.out/'summary.json', summary)
    metrics.close()
    (args.out/'COMPLETE').write_text(checkpoints['final']['sha256']+'\n')
    print(json.dumps(dict(arm=args.arm, final_step=step, ppl=final_score['ppl'],
                          training_elapsed_s=summary['training_elapsed_s'])), flush=True)


if __name__=='__main__':
    main()

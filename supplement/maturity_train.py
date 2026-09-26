"""Controlled pretraining-maturity branches using the validated mixed-head model."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanogpt.model import GPT, GPTConfig, fused_dispatch_count
from nanogpt.data_sampling import ShuffledNonoverlapSampler
from supplement.attention import install_packed
from supplement.train import evaluate, intervene
from supplement.maturity_protocol import completed, learning_rate, read, require, sha, tokens_per_update, verify_source, write


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def split_after_document(data, offset):
    for start in range(offset, len(data), 1 << 20):
        found = np.flatnonzero(data[start:start+(1 << 20)] == 50256)
        if len(found):
            return start + int(found[0]) + 1
    raise RuntimeError('No document boundary after the requested partition')


def prepare_data(source, root, manifest):
    spec = manifest['data']
    directory = Path(spec['directory'])
    registry = root/'data.json'
    if registry.exists():
        return read(registry)
    preparation = Path(__file__).resolve().parents[1] / 'data/prepare_fineweb_edu_scale.py'
    command = [sys.executable, str(preparation), '--out-dir', str(directory),
               '--revision', spec['revision'], '--train-tokens', str(spec['tokens']),
               '--validation-tokens', '10000000', '--workers', '32']
    subprocess.run(command, check=True)
    metadata = read(directory/'metadata.json')
    require(metadata['resolved_revision'] == spec['revision'] and metadata['tokenizer'] == 'tiktoken:gpt2',
            'Unexpected corpus revision or tokenizer')
    train = np.memmap(directory/'train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(directory/'val.bin', dtype=np.uint16, mode='r')
    boundary = split_after_document(train, spec['source_boundary'])
    val_boundary = split_after_document(val, len(val)//2)
    cfg = manifest['settings']
    require(boundary > max(cfg['parent_updates'])*tokens_per_update(cfg)+cfg['context'],
            'Source partition is too small')
    require(len(train)-boundary > cfg['clock_update_cap']*tokens_per_update(cfg)+cfg['context'],
            'Reserved recovery partition is too small')
    result = dict(directory=str(directory), source_end=boundary, calibration_end=val_boundary,
                  source_manifest_sha256=sha(source/'manifest.json'),
                  files={n: dict(bytes=(directory/n).stat().st_size, sha256=sha(directory/n))
                         for n in ('train.bin', 'val.bin', 'metadata.json')},
                  partition='Source/recovery and calibration/reporting split at end-of-document tokens; no target-window reuse within a run.')
    write(registry, result)
    return result


class Windows:
    def __init__(self, registry, cfg, device, recovery):
        self.cfg, self.device = cfg, device
        directory = Path(registry['directory'])
        self.train = np.memmap(directory/'train.bin', dtype=np.uint16, mode='r')
        self.val = np.memmap(directory/'val.bin', dtype=np.uint16, mode='r')
        lo, end = (registry['source_end'], len(self.train)) if recovery else (0, registry['source_end'])
        self.sampler = ShuffledNonoverlapSampler(lo=lo, end=end, window_size=cfg['context'],
                                                seed=cfg['seed']+(60003 if recovery else 30003))
        self.calibration_rng = torch.Generator().manual_seed(cfg['seed']+100003)
        self.half = registry['calibration_end']
        self.report_starts = list(range(self.half, len(self.val)-cfg['context'], cfg['context']))
        if cfg.get('test'):
            self.report_starts = self.report_starts[:2]
        require(self.report_starts and self.half > cfg['context'], 'Insufficient held-out data')

    def batch(self, data, starts):
        length = self.cfg['context']
        rows = np.stack([data[int(i):int(i)+length+1] for i in starts]).astype(np.int64)
        x, y = torch.from_numpy(rows[:, :-1].copy()), torch.from_numpy(rows[:, 1:].copy())
        if self.device == 'cuda':
            x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        return x, y

    def training(self):
        return self.batch(self.train, self.sampler.take(self.cfg['batch']))

    def calibration(self, _split):
        starts = torch.randint(self.half-self.cfg['context'], (self.cfg['calibration_batch'],),
                               generator=self.calibration_rng)
        return self.batch(self.val, starts)


def initial_pruning(model, optimiser, specification, cfg, device):
    if specification is None:
        return None
    heads = specification['heads']
    require(len(heads) == cfg['n_layer'], 'Initial pruning layer count differs')
    require(all(sorted(set(row)) == row and all(isinstance(h, int) and 0 <= h < cfg['n_head']
                for h in row) for row in heads), 'Invalid initial pruning head set')
    require(sum(map(len, heads)) == round(.25*cfg['n_layer']*cfg['n_head']),
            'Initial pruning must remove exactly 25% of heads')
    sync(device)
    started = time.perf_counter()
    install_packed(model, heads, 'prune', optimiser)
    sync(device)
    return dict(mode='prune', heads=heads, at_update=0,
                construction_s=time.perf_counter()-started, total_s=0.)


def load_parent(folder, cfg, device, model, optimiser, windows, recovery, expected_pruning=None):
    summary = read(folder/'summary.json')
    require(sha(folder/'final.pt') == summary['checkpoint_sha256'], 'Parent hash mismatch')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    if expected_pruning is None:
        require(state['intervention']['mode'] == 'ordinary', 'Parent must be ordinary')
    else:
        require(state['intervention']['mode'] == 'prune'
                and state['intervention'].get('at_update') == 0
                and state['intervention']['heads'] == expected_pruning['heads'],
                'Initial-pruning parent structure differs')
    require(state['model_config'] == asdict(model.config), 'Parent model configuration differs')
    require(state['settings'] == cfg, 'Parent settings differ')
    model.load_state_dict(state['model'], strict=True)
    optimiser.load_state_dict(state['optimizer'])
    if not recovery:
        windows.sampler.load_state_dict(state['sampler'])
        require(windows.sampler.cursor == state['step']*cfg['batch']*cfg['accumulation'], 'Parent data cursor mismatch')
    torch.set_rng_state(state['torch_rng'])
    if device == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'])
    return state['step'], summary


def train(source, root, name, device='cuda'):
    manifest = verify_source(source)
    cfg, spec = manifest['settings'], manifest['phases'][name]
    registry = read(root/'data.json')
    require(registry['source_manifest_sha256'] == sha(source/'manifest.json'), 'Data/source mismatch')
    input_root = root
    if 'external_reference' in manifest:
        from supplement.gate_pruning_protocol import validate_inputs
        input_root = validate_inputs(source, root, manifest)
    for filename, record in registry['files'].items():
        path = Path(registry['directory'])/filename
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Data changed')
    out = root/'runs'/name
    require(not out.exists(), 'Refusing to overwrite ' + str(out))
    out.mkdir(parents=True)
    write(out/'config.json', dict(settings=cfg, phase=spec, manifest_sha256=sha(source/'manifest.json')))
    started = time.perf_counter()
    torch.set_num_threads(2 if device=='cpu' else 8)
    torch.manual_seed(cfg['seed'])
    if device == 'cuda':
        require(torch.cuda.device_count() == 1, 'Expected one GPU')
        require(torch.cuda.is_bf16_supported(), 'A CUDA GPU with BF16 support is required')
        torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(GPTConfig(block_size=cfg['context'], n_layer=cfg['n_layer'], n_head=cfg['n_head'],
                          n_embd=cfg['n_embd'], vocab_size=cfg['vocab_size'], bias=False,
                          position_encoding='rope', rand_attn_dynamic=True,
                          rand_attn_prior_repr='decomposed', freeze_pattern_dtype='bfloat16')).to(device)
    optimiser = model.configure_optimizers(cfg['weight_decay'], cfg['peak_lr'], (cfg['beta1'],cfg['beta2']),device)
    pruning_spec = manifest.get('initial_pruning')
    require(pruning_spec is None or spec['mode']=='prune_from_start', 'Unexpected initial-pruning phase')
    initial = initial_pruning(model, optimiser, pruning_spec, cfg, device)
    recovery = spec['phase']=='recovery'
    windows = Windows(registry, cfg, device, recovery)
    parent_summary = None
    step = 0
    if spec['parent']:
        require(completed(input_root, spec['parent']), 'Parent has not completed')
        step, parent_summary = load_parent(input_root/'runs'/spec['parent'],cfg,device,model,optimiser,windows,recovery,
                                           expected_pruning=pruning_spec)
        require(parent_summary['data_sha256']==sha(input_root/'data.json'), 'Parent corpus differs')
    start_step = step
    context = (lambda: torch.autocast('cuda',dtype=torch.bfloat16)) if device=='cuda' else nullcontext
    elapsed = lambda: time.perf_counter()-started
    metrics = (out/'metrics.jsonl').open('w',buffering=1)
    def log(record):
        metrics.write(json.dumps(record,allow_nan=False)+'\n')
    def monitor(kind, complete=False):
        begin=elapsed()
        score=evaluate(model,windows,context,complete=complete)
        log(dict(event='evaluation',kind=kind,step=step,elapsed_s=elapsed(),evaluation_s=elapsed()-begin,**score))
        return score
    backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device=='cuda' else nullcontext()
    budget = None
    if spec.get('budget')=='clock':
        reference=read(input_root/'runs'/spec['reference']/'summary.json')
        require(reference['parent_sha256']==parent_summary['checkpoint_sha256']
                and reference['local_updates']==cfg['recovery_updates']
                and reference['settings']==cfg and reference['data_sha256']==sha(input_root/'data.json'),
                'Invalid equal-time reference')
        budget=reference['training_elapsed_s']
    snapshots={}
    intervention=initial or dict(mode='ordinary',heads=[[] for _ in model.transformer.h],total_s=0.)
    if initial is not None:
        intervention['new_removal_this_phase'] = spec['parent'] is None
        intervention['total_s'] = intervention['construction_s'] if spec['parent'] is None else 0.
    def save(label):
        sync(device)
        path=out/(label+'.pt')
        state=dict(model=model.state_dict(),optimizer=optimiser.state_dict(),model_config=asdict(model.config),
                   settings=cfg,step=step,sampler=windows.sampler.state_dict(),intervention=intervention,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None)
        temporary=path.with_suffix('.tmp')
        torch.save(state,temporary)
        temporary.replace(path)
        record=dict(path=str(path),sha256=sha(path),step=step,elapsed_s=elapsed())
        write(out/(label+'.receipt.json'),record)
        snapshots[label]=record
    local_step=0
    times=[]
    next_clock_eval=.1
    with backend:
        before=monitor('before_training' if initial is not None else 'before_intervention')
        if recovery and initial is None:
            if spec['mode'] == 'prune_gate_taylor':
                from supplement.gate_pruning import intervene as prune_by_gate
                intervention=prune_by_gate(model,optimiser,windows,context,cfg)
            else:
                intervention=intervene(model,optimiser,windows,context,dict(mode=spec['mode'],rate=.25),cfg['seed'],cfg)
        write(out/'intervention.json',intervention)
        after=monitor('after_intervention') if initial is None and spec['mode']!='ordinary' else before
        fixed={n:b.detach().cpu().clone() for n,b in model.named_buffers()
               if n.endswith(('.dyn_alpha','.dyn_rho','.dyn_z','.dyn_frozen','.dyn_decomp'))}
        if recovery:
            save('phase_start' if initial is not None else 'intervention')
        if device=='cuda':
            torch.cuda.reset_peak_memory_stats()
        model.train()
        end=spec.get('end_step',start_step+cfg['recovery_updates'])
        while True:
            if budget is None and step>=end:
                break
            if budget is not None and elapsed()+max(3., max(times[-10:],default=3.)*2)>=budget:
                break
            require(not recovery or local_step<cfg['clock_update_cap'], 'Recovery safety/data limit reached')
            lr=learning_rate(spec['phase'],step,local_step,cfg,elapsed()/budget if budget else None)
            for group in optimiser.param_groups:
                group['lr']=lr
            sync(device)
            begin=time.perf_counter()
            losses=[]
            for _ in range(cfg['accumulation']):
                x,y=windows.training()
                with context():
                    _,loss=model(x,y)
                (loss/cfg['accumulation']).backward()
                losses.append(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            value=float(torch.stack(losses).mean())
            require(math.isfinite(value) and torch.isfinite(norm).item(), 'Non-finite training update')
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            sync(device)
            times.append(time.perf_counter()-begin)
            step+=1
            local_step+=1
            require(budget is None or elapsed()<=budget, 'Update exceeded the wall-clock budget')
            log(dict(event='update',step=step,local_step=local_step,loss=value,lr=lr,gradient_norm=float(norm),
                     update_s=times[-1],elapsed_s=elapsed(),tokens=step*tokens_per_update(cfg)))
            if local_step%cfg['eval_every']==0:
                print(f'{name} step={step} loss={value:.5f} update_s={times[-1]:.3f}',flush=True)
            if budget is None:
                if local_step%cfg['eval_every']==0 and step<end:
                    monitor('recovery' if recovery else 'trajectory')
                interval=cfg['source_save_every']
                if step<end and ((recovery and local_step in cfg['recovery_save_updates'])
                                 or (not recovery and local_step%interval==0)):
                    save(f'update_{step:07d}')
            elif elapsed()/budget>=next_clock_eval and elapsed()+max(30.,times[-1]*15)<budget:
                monitor('recovery')
                next_clock_eval=(math.floor(elapsed()/budget*10)+1)/10
    training_s=elapsed()
    require(times,'No updates completed')
    for n,b in model.named_buffers():
        if n in fixed:
            require(torch.equal(b.detach().cpu(),fixed[n]),'A fixed prior changed during training')
    require(spec['mode']!='mean' or device!='cuda' or fused_dispatch_count()>0,'Mean did not use fused kernel')
    allocated=torch.cuda.max_memory_allocated() if device=='cuda' else 0
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device=='cuda' else nullcontext():
        final=monitor('final',complete=True)
    save('final')
    summary=dict(name=name,phase=spec,settings=cfg,step=step,local_updates=local_step,
                 parent_step=start_step,parent_sha256=parent_summary['checkpoint_sha256'] if parent_summary else None,
                 source_manifest_sha256=sha(source/'manifest.json'),data_sha256=sha(root/'data.json'),
                 checkpoint_sha256=snapshots['final']['sha256'],checkpoints=snapshots,intervention=intervention,
                 before=before,after=after,final_score=final,training_elapsed_s=training_s,budget_s=budget,
                 update_s=sum(times),mean_update_ms=1000*sum(times)/len(times),peak_allocated_bytes=allocated,
                 total_processed_tokens=step*tokens_per_update(cfg),post_intervention_tokens=local_step*tokens_per_update(cfg) if recovery else 0,
                 parameters=sum(p.numel() for p in model.parameters()),sampler=windows.sampler.state_dict(),
                 fused_dispatches=fused_dispatch_count(),host=os.uname().nodename,
                 torch=torch.__version__,cuda=torch.version.cuda,
                 gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                 process_total_s=elapsed(),timing_scope='After corpus integrity checks: includes model loading, intervention, monitoring and intermediate saves; excludes final evaluation/export.')
    if initial is not None:
        summary['initial_pruning'] = pruning_spec
        summary['post_intervention_tokens'] = step*tokens_per_update(cfg)
    if 'external_reference' in manifest:
        summary['external_reference'] = manifest['external_reference']
        summary['input_data_sha256'] = sha(input_root/'data.json')
    write(out/'summary.json',summary)
    (out/'COMPLETE').write_text(sha(out/'summary.json')+'\n')
    metrics.close()
    print(json.dumps(dict(name=name,ppl=final['ppl'],step=step,seconds=training_s)),flush=True)


def bundle(source,root,task_name):
    manifest=verify_source(source)
    task=next(t for t in manifest['tasks'] if t['name']==task_name)
    root.mkdir(parents=True,exist_ok=True)
    if task_name=='source_low':
        prepare_data(source,root,manifest)
    require((root/'data.json').exists(),'Data preparation has not completed')
    for name in task['phases']:
        env=os.environ.copy()
        for key,label in [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
            folder=root/'caches'/name/label
            folder.mkdir(parents=True,exist_ok=True)
            env[key]=str(folder)
        command=[sys.executable,'-m','supplement.maturity_train','worker','--source',str(source),
                 '--root',str(root),'--name',name]
        subprocess.run(command,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['bundle','worker'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    (bundle if args.action=='bundle' else train)(args.source.resolve(),args.root.resolve(),args.name)


if __name__=='__main__':
    main()

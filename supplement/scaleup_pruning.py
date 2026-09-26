"""Pack one 1B midpoint with gate-Taylor scores, preserving the resume contract."""

import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from supplement.gate_pruning import gate_derivatives, rank_heads
from supplement.maturity_protocol import require, sha, verify_source, write
from nanogpt.model import GPT, GPTConfig
from supplement.attention import install_packed


def load_model_optimiser(payload, device):
    model=GPT(GPTConfig(**payload['model_args']))
    layout=payload.get('attention_layout')
    if layout is not None:
        require(layout['mode']=='prune','Unexpected packed layout')
        install_packed(model,layout['heads'],'prune')
    model.load_state_dict(payload['model'],strict=True)
    model.to(device)
    cfg=payload['config']
    optimiser=model.configure_optimizers(cfg['weight_decay'],cfg['learning_rate'],
                                       (cfg['beta1'],cfg['beta2']),device)
    optimiser.load_state_dict(payload['optimizer'])
    return model,optimiser


def named_optimiser(model,optimiser):
    names={id(p):n for n,p in model.named_parameters()}
    require(all(id(p) in names for g in optimiser.param_groups for p in g['params']),
            'Optimiser contains orphan parameters')
    return [[names[id(p)] for p in g['params']] for g in optimiser.param_groups]


def pack_payload(payload,model,optimiser,heads):
    before=named_optimiser(model,optimiser)
    install_packed(model,heads,'prune',optimiser)
    require(before==named_optimiser(model,optimiser),'Packing changed optimiser parameter order')
    result=dict(payload)
    result.update(model=model.state_dict(),optimizer=optimiser.state_dict(),
                  attention_layout=dict(mode='prune',heads=heads),optimiser_parameter_names=before)
    return result


def compare_optimisers(a,b):
    require(a['param_groups']==b['param_groups'],'Optimiser groups changed on reload')
    require(a['state'].keys()==b['state'].keys(),'Optimiser state keys changed')
    for key,state in a['state'].items():
        require(state.keys()==b['state'][key].keys(),'Adam state fields changed')
        for name,value in state.items():
            other=b['state'][key][name]
            if torch.is_tensor(value):
                require(torch.equal(value.cpu(),other.cpu()),'Adam state changed: '+name)
            else:
                require(value==other,'Adam scalar changed')


def prepare(source,out):
    m=verify_source(source)
    parent=m['parent']
    require(not out.exists(),'Existing packed checkpoint directory')
    require(sha(parent['path'])==parent['sha256'],'Midpoint changed')
    out.mkdir(parents=True)
    started=time.perf_counter()
    state=torch.load(parent['path'],map_location='cpu',weights_only=False)
    require(state['iter_num']==18756 and state.get('attention_layout') is None,'Wrong ordinary midpoint')
    model,opt=load_model_optimiser(state,'cuda')
    require((model.config.n_layer,model.config.n_head,model.config.n_embd,model.config.block_size)==
            (32,16,1536,8192),'Wrong 1B architecture')
    require(all(not b.main_block.dyn_frozen.any() for b in model.transformer.h),'Parent already replaced')
    require(state['train_sampler_state']['world_size']==4,'Wrong source sampler world size')
    model.eval()
    train=np.memmap(m['data_dir']+'/train.bin',mode='r',dtype=np.uint16)
    rng=torch.Generator()
    rng.set_state(state['freeze_batch_rng_state'].cpu())
    starts,rows,digest=[],[],hashlib.sha256()
    context=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(16):
            start=int(torch.randint(0,len(train)-8192,(1,),generator=rng))
            starts.append(start)
            x=torch.from_numpy(train[start:start+8192].astype(np.int64))[None].cuda()
            y=torch.from_numpy(train[start+1:start+8193].astype(np.int64))[None].cuda()
            digest.update(x.cpu().numpy().tobytes()); digest.update(y.cpu().numpy().tobytes())
            rows.append(gate_derivatives(model,x,y,context).cpu().double())
    raw=torch.cat(rows).abs().mean(0)
    scores,heads=rank_heads(raw)
    require(sum(map(len,heads))==128,'Wrong prune count')
    scored=time.perf_counter()
    packed=pack_payload(state,model,opt,heads)
    packed['pruning_intervention']=dict(selector='head_gate_taylor_layer_l2',heads=heads,
        rate=.25,parent_sha256=parent['sha256'],calibration_starts=starts,
        calibration_data_sha256=digest.hexdigest(),signed_derivatives=torch.cat(rows).tolist(),
        raw_scores=raw.tolist(),normalised_scores=scores.tolist(),scoring_s=scored-started,
        calibration='16 train windows, same rank-0 freeze RNG and sampling rule as the mean intervention')
    packed['experiment']=dict(manifest_sha256=sha(source/'manifest.json'),
                              parent_checkpoint_sha256=parent['sha256'])
    path=out/'ckpt.pt'
    torch.save(packed,path.with_suffix('.pt.tmp'))
    path.with_suffix('.pt.tmp').replace(path)
    reloaded=torch.load(path,map_location='cpu',weights_only=False)
    check,check_opt=load_model_optimiser(reloaded,'cpu')
    require(named_optimiser(check,check_opt)==packed['optimiser_parameter_names'],'Reload changes parameter order')
    for name,value in model.state_dict().items():
        require(torch.equal(value.cpu(),check.state_dict()[name]),'Model reload changed '+name)
    compare_optimisers(opt.state_dict(),check_opt.state_dict())
    for key in ('iter_num','train_sampler_state','train_batch_rng_state','eval_batch_rng_state',
                'freeze_batch_rng_state','intervention_batch_rng_state','cpu_rng_state'):
        a,b=state[key],reloaded[key]
        require(torch.equal(a,b) if torch.is_tensor(a) else a==b,'Resume field changed: '+key)
    report=dict(**packed['pruning_intervention'],checkpoint_sha256=sha(path),
                preparation_s=time.perf_counter()-started,adam_reload_exact=True,
                training_resume_fields_unchanged=True,manifest_sha256=sha(source/'manifest.json'))
    write(out/'intervention.json',report)
    write(out/'PACKING_COMPLETE',report)

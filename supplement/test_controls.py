"""Numerical oracles for the new controls and the existing fused backward."""

import copy
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from nanogpt.model import GPT, GPTConfig, CausalSelfAttention, RotaryEmbedding, fused_dispatch_count
from supplement.attention import PackedAttention, CausalUniform, install_packed, matched_random_heads, random_eligible_heads
from supplement.protocol import token_lr, clock_lr, continuation_budget, settings


def reference(module, x, selected, mode, rope):
    b,t,c = x.shape
    shape = (b,t,module.n_head,module.head_dim)
    q,k,v = [proj(x).view(shape).transpose(1,2)
             for proj in (module.q_proj,module.k_proj,module.v_proj)]
    q,k = module._apply_rope(q,k,rope)
    mask = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
    weights = ((q.float()@k.float().transpose(-1,-2))/math.sqrt(module.head_dim)).masked_fill(~mask,-torch.inf).softmax(-1)
    uniform = mask.float()/torch.arange(1,t+1,device=x.device).view(-1,1)
    pieces=[]
    for h in range(module.n_head):
        if h in selected:
            if mode=='prune': pieces.append(v[:,h]*0)
            elif mode=='mean':
                prior = module._decomp_patterns(torch.tensor([h], device=x.device), t)[0].to(v.dtype)
                pieces.append(prior@v[:,h])
            else: pieces.append((uniform@v[:,h].float()).to(v.dtype))
        else: pieces.append(weights[:,h].to(v.dtype)@v[:,h])
    y=torch.stack(pieces,1).transpose(1,2).reshape(b,t,c)
    return module.c_proj(y)


def error(actual, expected):
    delta=(actual.float()-expected.float())
    return dict(max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                relative_l2=float(delta.norm()/expected.float().norm().clamp_min(1e-8)))


class Controls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(812)

    def test_prefix_sum_and_backward(self):
        for length in [1,7,129]:
            x=torch.randn(2,3,length,5,requires_grad=True)
            oracle=x.detach().clone().requires_grad_()
            matrix=torch.ones(length,length).tril()/torch.arange(1,length+1).view(-1,1)
            y=CausalUniform.apply(x); expected=matrix@oracle
            grad=torch.randn_like(y)
            y.backward(grad); expected.backward(grad)
            torch.testing.assert_close(y,expected,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(x.grad,oracle.grad,atol=1e-6,rtol=1e-5)

    def test_layer_oracles(self):
        for mode in ['uniform','prune']:
            for selected in [[],[1,3],list(range(4))]:
                for bias in [False,True]:
                    cfg=GPTConfig(block_size=17,n_embd=32,n_head=4,n_layer=1,
                                  bias=bias,position_encoding='rope')
                    original=CausalSelfAttention(cfg)
                    packed=PackedAttention(copy.deepcopy(original),selected,mode)
                    x=torch.randn(2,13,32,requires_grad=True)
                    ox=x.detach().clone().requires_grad_()
                    rope=RotaryEmbedding(8,17)(13,torch.float32)
                    y=packed(x,rope=rope); expected=reference(original,ox,selected,mode,rope)
                    grad=torch.randn_like(y)
                    y.backward(grad);expected.backward(grad)
                    torch.testing.assert_close(y,expected,atol=2e-6,rtol=2e-5)
                    torch.testing.assert_close(x.grad,ox.grad,atol=2e-6,rtol=2e-5)

    def test_adam_state_and_continuation(self):
        cfg=GPTConfig(block_size=8,vocab_size=64,n_embd=32,n_head=4,n_layer=2,
                      bias=False,position_encoding='rope')
        for mode in ['uniform','prune']:
            model=GPT(cfg); opt=model.configure_optimizers(.1,1e-3,(.9,.95),'cpu')
            x=torch.randint(0,64,(2,8));y=torch.randint(0,64,(2,8))
            model(x,y)[1].backward();opt.step();opt.zero_grad(set_to_none=True)
            old=model.transformer.h[0].main_block
            old_parameter=old.q_proj.weight
            old_state=copy.deepcopy(opt.state[old_parameter])
            old_weight=old_parameter.detach().clone()
            rng=torch.get_rng_state().clone()
            install_packed(model,[[1,3],[]],mode,opt)
            new=model.transformer.h[0].main_block
            rows=torch.tensor(list(range(8))+list(range(16,24)))
            torch.testing.assert_close(new.q_proj.weight,old_weight[rows])
            for key in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(opt.state[new.q_proj.weight][key],old_state[key][rows])
            torch.testing.assert_close(opt.state[new.q_proj.weight]['step'],old_state['step'])
            self.assertNotIn(old_parameter,opt.state)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            # A complete subsequent update must still change the retained weights.
            before=new.q_proj.weight.detach().clone()
            model(x,y)[1].backward();opt.step();opt.zero_grad(set_to_none=True)
            self.assertFalse(torch.equal(before,new.q_proj.weight))
            parameter_ids={id(p) for group in opt.param_groups for p in group['params']}
            self.assertEqual(parameter_ids,{id(p) for p in model.parameters() if p.requires_grad})
            # Round-trip a physically packed checkpoint.
            clone=GPT(cfg);clone_opt=clone.configure_optimizers(.1,1e-3,(.9,.95),'cpu')
            install_packed(clone,[[1,3],[]],mode,clone_opt)
            clone.load_state_dict(model.state_dict());clone_opt.load_state_dict(opt.state_dict())
            torch.testing.assert_close(model(x,y)[0],clone(x,y)[0])

    def test_random_layer_counts(self):
        reference_heads=[[0,1],[],[2,3,4],[0,1,2,3,4,5]]
        first=matched_random_heads(reference_heads,6,91)
        self.assertEqual(first,matched_random_heads(reference_heads,6,91))
        self.assertEqual(list(map(len,reference_heads)),list(map(len,first)))
        self.assertNotEqual(first,matched_random_heads(reference_heads,6,92))

    def test_budget_and_schedule(self):
        cfg=settings()
        self.assertEqual(continuation_budget(100,40),60)
        with self.assertRaises(ValueError): continuation_budget(30,40)
        self.assertEqual(clock_lr(2500,0,60,cfg),token_lr(2500,cfg))
        self.assertEqual(clock_lr(2500,60,60,cfg),cfg['min_lr'])
        self.assertGreater(clock_lr(2500,1,60,cfg),clock_lr(2500,50,60,cfg))

    def test_random_common_fit_gate(self):
        reference_heads=[[0,1],[],[0,1,2]]
        def accept(pairs):return [(li,h) for li,h in pairs if h not in [0,2]]
        selected=random_eligible_heads(reference_heads,6,91,accept)
        self.assertEqual(list(map(len,selected)),list(map(len,reference_heads)))
        self.assertTrue(all(h not in [0,2] for heads in selected for h in heads))
        self.assertEqual(selected,random_eligible_heads(reference_heads,6,91,accept))
        with self.assertRaises(RuntimeError):
            random_eligible_heads([[0,1,2,3,4]],6,91,accept)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA validation runs in the GPU pilot')
    def test_cuda_gradients(self):
        records=[]
        for dtype in [torch.float32,torch.bfloat16]:
            for mode in ['uniform','prune','mean']:
                cfg=GPTConfig(block_size=192,n_embd=256,n_head=4,n_layer=1,bias=False,
                              position_encoding='rope',rand_attn_dynamic=True,
                              rand_attn_prior_repr='decomposed',freeze_pattern_dtype='bfloat16')
                original=CausalSelfAttention(cfg).cuda()
                if mode=='mean':
                    alpha=torch.randn(2,192,device='cuda')*.1
                    rho=torch.randn(2,192,device='cuda')*.1
                    original.freeze_heads_decomposed([1,3],alpha,rho)
                candidate=copy.deepcopy(original) if mode=='mean' else PackedAttention(copy.deepcopy(original),[1,3],mode)
                x=torch.randn(2,129,256,device='cuda',requires_grad=True)
                ox=x.detach().clone().requires_grad_()
                rope=RotaryEmbedding(64,129).cuda()(129,dtype)
                context=(lambda:torch.autocast('cuda',dtype=dtype)) if dtype==torch.bfloat16 else nullcontext
                captured={}
                def hook(label):
                    def record(module, inputs, output):
                        output.retain_grad()
                        captured[label]=output
                    return record
                candidate.v_proj.register_forward_hook(hook('candidate_v'))
                original.v_proj.register_forward_hook(hook('reference_v'))
                before=fused_dispatch_count()
                with context():
                    y=candidate(x,rope=rope)
                    target=reference(original,ox,[1,3],mode,rope)
                if mode=='mean': self.assertGreater(fused_dispatch_count(),before)
                grad=torch.randn_like(y)
                y.backward(grad);target.backward(grad)
                pairs={'output':(y,target),'input_gradient':(x.grad,ox.grad)}
                live=torch.tensor(list(range(64))+list(range(128,192)),device='cuda')
                all_cols=torch.cat([live,torch.tensor(list(range(64,128))+list(range(192,256)),device='cuda')])
                reference_v_grad=captured['reference_v'].grad
                if mode!='mean':
                    reference_v_grad=reference_v_grad.index_select(-1,all_cols if mode=='uniform' else live)
                pairs['value_gradient']=(captured['candidate_v'].grad,reference_v_grad)
                for name in ['q_proj','k_proj','v_proj','c_proj']:
                    got=getattr(candidate,name).weight.grad
                    wanted=getattr(original,name).weight.grad
                    if mode!='mean':
                        if name in ['q_proj','k_proj']:wanted=wanted.index_select(0,live)
                        elif name=='v_proj':wanted=wanted.index_select(0,all_cols if mode=='uniform' else live)
                        else:wanted=wanted.index_select(1,all_cols if mode=='uniform' else live)
                    pairs[name+'_gradient']=(got,wanted)
                for name,(got,wanted) in pairs.items():
                    stats=error(got,wanted)
                    records.append(dict(dtype=str(dtype),mode=mode,component=name,**stats))
                    self.assertLess(stats['relative_l2'],.035 if dtype==torch.bfloat16 else .003,
                                    msg=str(records[-1]))
        path=os.environ.get('GRADIENT_REPORT')
        if path:Path(path).write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':unittest.main()

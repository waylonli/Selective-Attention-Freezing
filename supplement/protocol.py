"""Scientific settings and budget arithmetic, independent of the training loop."""

import math


def token_lr(step, settings):
    warmup, end = settings['warmup'], settings['updates']
    peak, minimum = settings['peak_lr'], settings['min_lr']
    if step < warmup:
        return peak * step / warmup
    fraction = min(1., max(0., (step - warmup) / (end - warmup)))
    return minimum + (peak - minimum) * .5 * (1 + math.cos(math.pi * fraction))


def clock_lr(start_step, elapsed, budget, settings):
    if budget <= 0:
        raise ValueError('continuation budget must be positive')
    virtual_step = start_step + (settings['updates'] - start_step) * min(1., elapsed / budget)
    return token_lr(virtual_step, settings)


def continuation_budget(total, prefix):
    remaining = total - prefix
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError('ordinary total must exceed the recorded prefix cost')
    return remaining


def arm_grid():
    arms = [{'name': 'ordinary_tokens', 'mode': 'ordinary', 'rate': 0., 'budget': 'tokens', 'parent': None}]
    for mode in ['mean', 'uniform', 'prune', 'random_mean']:
        for rate in [.25, .5]:
            arms.append({'name': f'{mode}{int(rate*100)}_tokens', 'mode': mode,
                         'rate': rate, 'budget': 'tokens', 'parent': 'midpoint'})
    arms.append({'name': 'mean25_early_tokens', 'mode': 'mean', 'rate': .25,
                 'budget': 'tokens', 'parent': 'quarter'})
    for mode, rate in [('ordinary', 0.), ('mean', .25), ('mean', .5),
                       ('uniform', .25), ('uniform', .5), ('prune', .25), ('prune', .5)]:
        arms.append({'name': f'{mode}{int(rate*100)}_clock', 'mode': mode,
                     'rate': rate, 'budget': 'clock', 'parent': 'midpoint'})
    return arms


def settings(pilot=False):
    return dict(context=4096, n_layer=12, n_head=12, n_embd=768, vocab_size=50304,
                batch=8, accumulation=15, updates=5000, warmup=500,
                peak_lr=6e-4, min_lr=6e-5, weight_decay=.1, beta1=.9, beta2=.95,
                grad_clip=1., calibration_batches=16, calibration_batch=2,
                fit_steps=400, fit_max_kl=.2, monitor_windows=32,
                eval_every=250, seeds=[1337,1338,1339],
                pilot=pilot)

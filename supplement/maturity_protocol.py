"""Training-budget study: explicit data, learning-rate and task contracts."""

import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tokens_per_update(cfg):
    return cfg['context'] * cfg['batch'] * cfg['accumulation']


def learning_rate(phase, global_step, local_step, cfg, clock_fraction=None):
    if phase == 'stable':
        return cfg['peak_lr'] * min(1., (global_step + 1) / cfg['warmup'])
    length = cfg['cooldown_updates'] if phase == 'cooldown' else cfg['recovery_updates']
    peak = cfg['peak_lr'] if phase == 'cooldown' else cfg['min_lr']
    minimum = cfg['min_lr'] if phase == 'cooldown' else cfg['recovery_min_lr']
    fraction = local_step / max(1, length - 1) if clock_fraction is None else clock_fraction
    require(math.isfinite(fraction), 'Non-finite schedule progress')
    fraction = min(1., max(0., fraction))
    return minimum + (peak - minimum) * .5 * (1 + math.cos(math.pi*fraction))


def graph(cfg):
    tasks, phases = [], {}
    previous = None
    for label, parent_updates in zip(('low', 'mid', 'high'), cfg['parent_updates']):
        stable, parent = 'stable_' + label, 'parent_' + label
        phases[stable] = dict(phase='stable', parent=previous, mode='ordinary',
                              end_step=parent_updates-cfg['cooldown_updates'])
        phases[parent] = dict(phase='cooldown', parent=stable, mode='ordinary', end_step=parent_updates)
        tasks.append(dict(name='source_' + label, phases=[stable, parent], needs=[previous] if previous else [],
                          hours={'low': 10, 'mid': 12, 'high': 16}[label], cpu=32 if label=='low' else 8))
        for mode in ('ordinary', 'mean', 'prune'):
            name = 'recover_' + label + '_' + mode
            phases[name] = dict(phase='recovery', parent=parent, mode=mode, budget='tokens')
            tasks.append(dict(name=name, phases=[name], needs=[parent], hours=4, cpu=8))
        if label in ('low', 'high'):
            for mode in ('ordinary', 'mean', 'prune'):
                name = 'clock_' + label + '_' + mode
                reference = 'recover_' + label + '_ordinary'
                phases[name] = dict(phase='recovery', parent=parent, mode=mode, budget='clock', reference=reference)
                tasks.append(dict(name=name, phases=[name], needs=[parent, reference], hours=4, cpu=8, clock=True))
        previous = stable
    return tasks, phases


def verify_source(source):
    manifest = read(source/'manifest.json')
    for name, digest in manifest['source_hashes'].items():
        require(sha(source/name) == digest, 'Source changed: ' + name)
    return manifest


def completed(root, name):
    folder = root/'runs'/name
    if not (folder/'COMPLETE').exists():
        return False
    require((folder/'COMPLETE').read_text().strip() == sha(folder/'summary.json'), 'Invalid completion: ' + name)
    summary = read(folder/'summary.json')
    require(sha(folder/'final.pt') == summary['checkpoint_sha256'], 'Checkpoint changed: ' + name)
    return True


def ready_tasks(manifest, root, include_clock=False):
    return [t for t in manifest['tasks'] if (include_clock or not t.get('clock'))
            and not (root/'submissions'/(t['name']+'.json')).exists()
            and all(completed(root, p) for p in t['needs'])]

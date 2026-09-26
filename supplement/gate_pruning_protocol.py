"""Keep pruning-specific selector inputs separate from matched-head controls."""

from pathlib import Path

from supplement.maturity_protocol import completed, read, require, sha


def validate_inputs(source, root, manifest, check_checkpoints=False):
    ref = manifest['external_reference']
    original = Path(ref['root'])
    require(original.resolve() != root.resolve(), 'Control results must have their own directory')
    require(sha(original/'data.json') == ref['data_sha256'], 'Reference data registry changed')
    old, new = read(original/'data.json'), read(root/'data.json')
    require(old['source_manifest_sha256'] == ref['manifest_sha256'], 'Reference manifest differs')
    require(new['source_manifest_sha256'] == sha(source/'manifest.json'), 'New manifest differs')
    require({k:v for k,v in old.items() if k != 'source_manifest_sha256'} ==
            {k:v for k,v in new.items() if k != 'source_manifest_sha256'}, 'Corpus partitions differ')
    summaries = {}
    for name, expected in ref['inputs'].items():
        folder = original/'runs'/name
        require(sha(folder/'summary.json') == expected['summary_sha256']
                and (folder/'COMPLETE').read_text().strip() == expected['summary_sha256'],
                'Reference completion changed: ' + name)
        summary = read(folder/'summary.json')
        require(summary['checkpoint_sha256'] == expected['checkpoint_sha256']
                and summary['settings'] == manifest['settings']
                and summary['source_manifest_sha256'] == ref['manifest_sha256']
                and summary['data_sha256'] == ref['data_sha256'], 'Reference provenance differs')
        if check_checkpoints:
            require(completed(original, name), 'Reference checkpoint unavailable: ' + name)
        summaries[name] = summary
    parent = summaries['parent_low']
    require(parent['step'] == manifest['settings']['parent_updates'][0]
            and parent['intervention']['mode'] == 'ordinary', 'Invalid source parent')
    if any(p['budget'] == 'clock' for p in manifest['phases'].values()):
        ordinary = summaries.get('recover_low_ordinary', {})
        require(ordinary.get('parent_sha256') == parent['checkpoint_sha256']
                and ordinary.get('phase', {}).get('budget') == 'tokens'
                and ordinary.get('phase', {}).get('mode') == 'ordinary'
                and ordinary.get('local_updates') == manifest['settings']['recovery_updates']
                and ordinary.get('training_elapsed_s') == ref['clock_budget_s'], 'Invalid clock reference')
    else:
        require(set(summaries) == {'parent_low'} and ref['clock_budget_s'] is None,
                'A token-only control must not invent a clock reference')
    require(all(p['mode'] == 'prune_gate_taylor' and p['phase'] == 'recovery'
                and p['parent'] == 'parent_low' for p in manifest['phases'].values()),
            'External inputs are restricted to gate-pruning recoveries')
    return original

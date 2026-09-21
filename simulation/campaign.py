"""Saved paired runs and the fixed experiment design used by the manuscript."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / 'data/runs'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_record(value, case, seed):
    if (not value.get('complete') or value.get('seed') != seed
            or value.get('identity') != case['identities'][str(seed)]):
        raise ValueError(f"Incomplete or incompatible result: {case['name']} seed {seed}")
    raw = value['raw']
    start = value['identity'].get('start_year', 2009)
    if raw['years'] != list(range(start, start + case['years'])):
        raise ValueError('Unexpected simulation horizon')
    for suffix in ('unc', 'cap'):
        series = [raw[f'annual_{field}_{suffix}'] for field in
                  ('conversions', 'spouse_conversions', 'children_saved', 'visas_consumed')]
        if any(len(values) != case['years'] for values in series):
            raise ValueError('Incomplete annual family series')
        if any(p + s + c != total for p, s, c, total in zip(*series)):
            raise ValueError('Family visa accounting failed')
    if case['role'] == 'cohorts' and any(
            row['still_dependent'] for row in value['cohorts'] if row['entry_year'] <= 2040):
        raise ValueError('Follow-up contains unresolved selected child cohorts')


def load_campaign(path=None, *, allow_partial_preview=False):
    """Read complete saved results, preserving their original scientific identities."""
    if allow_partial_preview:
        raise ValueError('The manuscript requires the complete declared experiment')
    path = Path(path or DEFAULT_RUNS).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    required = {'plan.json', 'paired_runs.jsonl.gz'}
    if set(manifest['files']) != required:
        raise ValueError('Saved-run manifest must identify the plan and paired records')
    for name, expected in manifest['files'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Saved result changed: {name}')
    plan = json.loads((path / 'plan.json').read_text())
    cases = {case['name']: case for case in plan['cases']}
    if len(cases) != len(plan['cases']):
        raise ValueError('Duplicate experiment name')
    expected = {(case['name'], seed) for case in cases.values() for seed in case['seeds']}
    if (sum(case['pairs'] for case in cases.values()) != len(expected)
            or any(len(set(case['seeds'])) != case['pairs'] for case in cases.values())
            or len(expected) != plan['planned_pairs'] or len(expected) != manifest['pairs']):
        raise ValueError('Declared sample size is inconsistent')
    records, seen = {name: [] for name in cases}, set()
    with gzip.open(path / 'paired_runs.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            name, value = item['case'], item['record']
            key = (name, value['seed'])
            if key not in expected or key in seen:
                raise ValueError('Unexpected or repeated experiment seed')
            validate_record(value, cases[name], key[1])
            if record_digest(value) != manifest['records'].get(f'{name}/{key[1]}'):
                raise ValueError('Paired record changed')
            records[name].append(value)
            seen.add(key)
    if seen != expected or len(manifest['records']) != len(expected):
        raise ValueError('Missing declared paired results')
    for values in records.values():
        values.sort(key=lambda value: value['seed'])
    hashes = {**manifest['files'], 'manifest.json': digest(path / 'manifest.json')}
    return plan, records, [], hashes


def experiment_cases(config, observed):
    cases = []
    for row in config['cases']:
        role = row['role']
        if role not in ('primary', 'sensitivity'):
            raise ValueError('Unknown experiment role')
        cases.append({'name': row['name'], 'role': role, 'years': 32,
                      'pairs': config[role + '_pairs'], 'overrides': {
                          'exit_specification': config['exit_specification'],
                          'catchall_median_age': row['median_age'],
                          'catchall_near_zero_age': row['near_zero_age'],
                          'catchall_tail_probability': config['catchall_tail_probability'],
                          'catchall_categories': config['catchall_categories']}})
    primary = [case for case in cases if case['role'] == 'primary']
    if len(primary) != 1 or primary[0]['name'] != config['primary_case']:
        raise ValueError('Declare exactly one primary experiment')
    overrides = primary[0]['overrides']
    cases += [{'name': 'cohort_followup', 'role': 'cohorts', 'years': 53,
               'pairs': config['cohort_pairs'], 'overrides': dict(overrides)},
              {'name': 'historical_replay', 'role': 'replay', 'years': 32,
               'pairs': config['replay_pairs'], 'overrides': {
                   **overrides, 'historical_allocation_path': str(observed)}}]
    if len({case['name'] for case in cases}) != len(cases):
        raise ValueError('Duplicate experiment name')
    for case in cases:
        if type(case['pairs']) is not int or case['pairs'] < 1:
            raise ValueError('Each experiment needs a positive fixed sample size')
        if not case['name'].replace('_', '').isalnum():
            raise ValueError('Unsafe experiment name')
        case['seeds'] = list(range(config['seed'], config['seed'] + case['pairs']))
    return cases


def _save_records(output, plan):
    from .experiments import atomic_json
    hashes = {}
    target = output / 'paired_runs.jsonl.gz'
    temporary = target.with_suffix('.tmp')
    with gzip.GzipFile(filename=str(temporary), mode='wb', compresslevel=6, mtime=0) as stream:
        for case in plan['cases']:
            for seed in case['seeds']:
                file = output / '.checkpoints' / case['name'] / 'runs' / f'seed_{seed}.json'
                value = json.loads(file.read_text())
                validate_record(value, case, seed)
                hashes[f"{case['name']}/{seed}"] = record_digest(value)
                line = json.dumps({'case': case['name'], 'record': value},
                                  separators=(',', ':'), allow_nan=False)
                stream.write(line.encode() + b'\n')
    temporary.replace(target)
    atomic_json(output / 'manifest.json', {
        'schema_version': 1, 'pairs': plan['planned_pairs'], 'records': hashes,
        'files': {name: digest(output / name) for name in ('plan.json', target.name)}})


def run_campaign(output, workers=2, config_path=None):
    """Run or resume the declared cases; never mix sources or change sample sizes."""
    import fcntl
    from .experiments import _run_job, atomic_json, source_fingerprint
    if not 1 <= workers <= 3:
        raise ValueError('Use one to three workers')
    output = Path(output).resolve()
    if output == DEFAULT_RUNS.resolve():
        raise ValueError('Use a separate directory; the published runs are preserved')
    config_path = Path(config_path or ROOT / 'data/experiment.json')
    config = json.loads(config_path.read_text())
    observed = ROOT / 'data/validation/dos_visa_consumption_reconciled.csv'
    fingerprint = source_fingerprint()
    cases = experiment_cases(config, observed)
    for case in cases:
        identity = dict(fingerprint)
        if case['role'] == 'replay':
            identity['historical_allocation_sha256'] = digest(observed)
        case['identities'] = {str(seed): {
            **identity, 'seed': seed, 'start_year': config['start_year'], 'years': case['years'],
            'overrides': case['overrides'], 'policy_branch_year': 2025} for seed in case['seeds']}
    plan = {'schema_version': 1, 'configuration': config, 'cases': cases,
            'planned_pairs': sum(case['pairs'] for case in cases),
            'source_fingerprint': fingerprint, 'observed_sha256': digest(observed)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another campaign is using this output directory') from exc
        if (output / 'plan.json').exists():
            if json.loads((output / 'plan.json').read_text()) != plan:
                raise ValueError('Sources, inputs or experiment settings changed; use a new directory')
        else:
            if any(p.name != '.run.lock' for p in output.iterdir()):
                raise ValueError('New campaign directory must be empty')
            atomic_json(output / 'plan.json', plan)
        if (output / 'manifest.json').exists():
            load_campaign(output)
            return plan
        jobs = [{'seed': seed, 'years': case['years'], 'start_year': config['start_year'],
                 'output': str(output / '.checkpoints' / case['name']),
                 'overrides': case['overrides'], 'identity': case['identities'][str(seed)]}
                for case in cases for seed in case['seeds']]
        if workers == 1:
            for index, job in enumerate(jobs, 1):
                _run_job(job)
                print(f"{index}/{len(jobs)} pairs complete", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(_run_job, job) for job in jobs]
                for index, future in enumerate(as_completed(futures), 1):
                    future.result()
                    print(f"{index}/{len(jobs)} pairs complete", flush=True)
        if source_fingerprint() != fingerprint or digest(observed) != plan['observed_sha256']:
            raise RuntimeError('Sources or inputs changed during execution; results were not finalized')
        _save_records(output, plan)
        load_campaign(output)
    return plan

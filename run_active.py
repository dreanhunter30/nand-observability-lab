#!/usr/bin/env python3
"""Reproducible CPU benchmark. All inference is Python standard library only."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import active_lab as lab

HERE = Path(__file__).resolve().parent
HARD_FAMILIES = ('constant_blocks', 'cloned_sources', 'deep_chain', 'reconvergent')
METHODS = ('passive_counts', 'single_full_access', 'pair_full_access',
           'single_half_access', 'pair_half_access', 'isolated_full_access')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def packed_digest(waves):
    return digest({k: hex(v) for k, v in waves.items()})


def encode_domains(domains):
    return {o: sorted(pairs) for o, pairs in sorted(domains.items())}


def case_specs():
    for cohort, offset in (('original_50', 0), ('new_50', 1000000)):
        for family_index, family in enumerate(lab.base.FAMILIES):
            for replicate in range(10):
                yield cohort, family, 2026101700 + family_index * 100 + replicate + offset
    for family_index, family in enumerate(HARD_FAMILIES):
        for replicate in range(5):
            yield 'hard_20', family, 2029000000 + family_index * 100 + replicate


def make_case(family, seed):
    return lab.make_hard(family, seed) if family in HARD_FAMILIES else lab.base.generate(family, seed)


def evaluate_graph(graph, primary, n):
    """Replay a recovered graph, independent of the hidden generator order."""
    remaining, values, mask = dict(graph), dict(primary), (1 << n) - 1
    while remaining:
        ready = [out for out, pair in remaining.items() if all(x in values for x in pair)]
        if not ready:
            raise ValueError('Unknown parents or directed cycle')
        for out in ready:
            x, y = remaining.pop(out)
            values[out] = mask ^ (values[x] & values[y])
    return values


def verify_full(domains, truth, inputs, node_class, counts, primary, expected, n):
    stats = lab.stats(domains, truth)
    if stats['whole_exact']:
        graph = {out: next(iter(pairs)) for out, pairs in domains.items()}
        assert lab.passive.validate_graph(graph, inputs, counts, node_class)
        assert evaluate_graph(graph, primary, n) == expected
        return True
    return False


def run_case(cohort, family, seed):
    c = make_case(family, seed)
    mapping, truth = lab.base.anonymous(c, seed + 200000)
    inputs, outputs = set(c.inputs), set(truth)
    train, holdout = lab.stimulus_sets(c, seed)
    assert not set(train) & set(holdout)
    waves = {mapping[k]: v for k, v in lab.base.evaluate_packed(
        c, lab.base.primary_waves(c, train), len(train)).items()}
    expected = {mapping[k]: v for k, v in lab.base.evaluate_packed(
        c, lab.base.primary_waves(c, holdout), len(holdout)).items()}
    primary = {k: expected[k] for k in c.inputs}
    devices = lab.passive.generate_unowned_mos_records(truth, waves, len(train), seed + 610000)
    node_class, counts, classes = lab.passive.infer_observable_counts(waves, devices, len(train))
    start = time.perf_counter()
    candidates = lab.base.infer_exact(waves, inputs, len(train))
    candidates, feasible = lab.base.prune_acyclic(candidates, inputs)
    assert feasible
    passive, info = lab.passive.propagate_counts(candidates, inputs, node_class, counts)
    assert info['consistent']
    passive_seconds = time.perf_counter() - start
    domains = {'passive_counts': passive}
    rows, schedules = [], {}
    common = {'cohort': cohort, 'family': family, 'seed': seed}

    def record(method, current, queries, seconds, access, max_clamps, total_queries=None):
        metrics = lab.stats(current, truth)
        assert metrics['wrong_unique'] == metrics['no_match'] == 0
        assert metrics['true_pair_in_candidates'] == len(truth)
        replay = verify_full(current, truth, inputs, node_class, counts, primary, expected, len(holdout))
        rows.append({**common, 'method': method, **metrics, 'incremental_queries': queries,
                     'cumulative_queries': queries if total_queries is None else total_queries,
                     'inference_and_simulation_seconds': seconds, 'available_internal_actuators': access,
                     'maximum_simultaneous_internal_clamps': max_clamps,
                     'holdout_full_graph_replay': replay})

    record('passive_counts', passive, 0, passive_seconds, 0, 0)
    for access_name, allowed in (('full_access', outputs), ('half_access', set(
            random.Random(seed + 730000).sample(sorted(outputs), len(outputs) // 2)))):
        current = passive
        total_queries = 0
        schedules[access_name] = {'allowed_internal_ids': sorted(allowed)}
        for intervention in ('single', 'pair'):
            stage_start = time.perf_counter()
            if intervention == 'single':
                scenarios = lab.single_plan(inputs, outputs, allowed, seed + 710000,
                                            forbidden_external=holdout)
            else:
                scenarios = lab.pair_plan(current, inputs, allowed, seed + 720000,
                                         forbidden_external=holdout)
            assert all(lab.external_assignment(a, commands, c.inputs) not in set(holdout)
                       for a, commands in scenarios)
            observed, clamped, n = lab.simulate_batch(c, mapping, scenarios)
            if n:
                current, info = lab.refine(current, observed, clamped, n, inputs, node_class, counts)
                assert info['consistent']
            method = f'{intervention}_{access_name}'
            domains[method] = current
            max_clamps = max((len(set(name for name, _ in overrides) & outputs)
                              for _, overrides in scenarios), default=0)
            total_queries += n
            record(method, current, n, time.perf_counter() - stage_start,
                   len(allowed), max_clamps, total_queries)
            schedules[access_name][intervention] = {
                'seed': seed + (710000 if intervention == 'single' else 720000),
                'queries': n, 'schedule_sha256': digest(scenarios),
                'observations_sha256': packed_digest(observed)}
    start = time.perf_counter()
    responses, descriptors, width, query_count, mismatches = lab.isolated_observations(c, mapping)
    graph, certificate = lab.decode_isolated(responses, descriptors, width, mismatches)
    assert graph == truth and certificate['certified']
    domains['isolated_full_access'] = {o: {pair} for o, pair in graph.items()}
    record('isolated_full_access', domains['isolated_full_access'], query_count,
           time.perf_counter() - start, len(outputs), max(0, len(outputs) - 1))
    evidence = {**common, 'input_names': c.inputs, 'training_inputs': train,
                'holdout_inputs': holdout, 'passive_waveforms_sha256': packed_digest(waves),
                'anonymized_true_pairs_FOR_SCORING_ONLY': truth, 'node_class': node_class,
                'observable_control_pin_counts': counts,
                'observable_class_count': len(classes), 'MOS_record_count': len(devices),
                'domains': {k: encode_domains(v) for k, v in domains.items()}, 'schedules': schedules,
                'isolated': {'width': width, 'queries': query_count, 'readback_mismatch_scenarios': mismatches,
                             'responses_hex': {o: hex(v) for o, v in responses.items()},
                             'certificate': certificate}}
    return rows, evidence, (responses, descriptors, width, truth)


def aggregate(rows):
    answer = {}
    for method in sorted({row['method'] for row in rows}):
        group = [row for row in rows if row['method'] == method]
        total_gates = sum(row['gates'] for row in group)
        correct = sum(row['correct_unique'] for row in group)
        answer[method] = {'circuits': len(group), 'gates': total_gates,
                         'correct_unique': correct, 'fraction_correct_unique': correct / total_gates,
                         'whole_exact': sum(row['whole_exact'] for row in group),
                         'wrong_unique': sum(row['wrong_unique'] for row in group),
                         'ambiguous': sum(row['ambiguous'] for row in group),
                         'no_match': sum(row['no_match'] for row in group),
                         'incremental_queries': sum(row['incremental_queries'] for row in group),
                         'cumulative_queries': sum(row['cumulative_queries'] for row in group),
                         'seconds': sum(row['inference_and_simulation_seconds'] for row in group)}
    return answer


def make_scale(size, seed):
    c = lab.base.Circuit(16)
    rng = random.Random(seed)
    available = list(c.inputs)
    for _ in range(size):
        available.append(c.nand(rng.choice(available), rng.choice(available)))
    c.outputs = available[-8:]
    return c


def run_scale(size):
    seed = 2030000000 + size
    c = make_scale(size, seed)
    mapping, truth = lab.base.anonymous(c, seed + 200000)
    begin = time.perf_counter()
    responses, descriptors, width, queries, mismatch = lab.isolated_observations(c, mapping)
    recovered, certificate = lab.decode_isolated(responses, descriptors, width, mismatch)
    assert recovered == truth and certificate['certified']
    cases = random.Random(seed + 1).sample(range(1 << len(c.inputs)), 256)
    primary = lab.base.primary_waves(c, cases)
    expected = {mapping[k]: v for k, v in lab.base.evaluate_packed(c, primary, len(cases)).items()}
    assert evaluate_graph(recovered, primary, len(cases)) == expected
    row = {'gates': size, 'inputs': len(c.inputs), 'seed': seed, 'queries': queries,
           'maximum_simultaneous_internal_clamps': size - 1,
           'seconds': time.perf_counter() - begin, 'whole_exact': recovered == truth,
           'holdout_full_graph_replay': True, 'certificate': certificate,
           'anonymized_true_pairs_FOR_SCORING_ONLY': truth,
           'responses_hex': {o: hex(v) for o, v in responses.items()}, 'width': width}
    return row, (responses, descriptors, width, truth)


def noise_benchmark(cached):
    total_reads = sum(width * len(truth) for _, _, width, truth in cached)
    results = []
    for p in (0.01, 0.05, 0.10):
        calibrated_r, calibrated_q = lab.calibrated_repeats(total_reads, p)
        for mode, repeats, effective_p in (('single_read', 1, p),
                                           ('calibrated_majority', calibrated_r, calibrated_q)):
            rng = random.Random(2031000000 + round(p * 10000) + repeats)
            correct = wrong = rejected = full = accepted = changed = 0
            for responses, descriptors, width, truth in cached:
                noisy = {}
                for output, response in sorted(responses.items()):
                    flips = lab.noise_mask(width, effective_p, rng)
                    changed += flips.bit_count()
                    noisy[output] = response ^ flips
                recovered, cert = lab.decode_isolated(noisy, descriptors, width)
                correct += sum(pair == truth[o] for o, pair in recovered.items())
                wrong += sum(pair != truth[o] for o, pair in recovered.items())
                rejected += len(truth) - len(recovered)
                accepted += cert['certified']
                full += recovered == truth
            results.append({'physical_bit_error_probability': p, 'mode': mode, 'repeats': repeats,
                            'effective_logical_bit_error_probability': effective_p,
                            'logical_queries': total_reads, 'raw_output_readouts': total_reads * repeats,
                            'family_wise_union_bound': min(1.0, total_reads * effective_p),
                            'required_bound': 1e-6 if mode == 'calibrated_majority' else None,
                            'circuits': len(cached), 'gates': sum(len(t) for _, _, _, t in cached),
                            'correct_decoded_pairs': correct, 'wrong_decoded_pairs': wrong,
                            'rejected_gates': rejected, 'whole_exact_circuits': full,
                            'locally_model_consistent_circuits': accepted,
                            'effective_readout_errors_sampled': changed,
                            'simulation': 'exact equivalent IID channel after majority; no explicit repeated raw draws',
                            'assumptions': 'known IID p; perfect actuation, readback, and stable NAND response'})
    return {'scope': '120 main circuits plus both scale references', 'total_logical_queries': total_reads,
            'cases': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    start = time.perf_counter()
    rows, evidence, cached = [], [], []
    specs = list(case_specs())
    for index, (cohort, family, seed) in enumerate(specs, 1):
        current_rows, current_evidence, isolation = run_case(cohort, family, seed)
        rows.extend(current_rows)
        evidence.append(current_evidence)
        cached.append(isolation)
        stage_metrics = {r['method']: r['correct_unique'] for r in current_rows}
        print(f'{index:03d}/{len(specs)} {cohort} {family} {seed}: {stage_metrics}', flush=True)
    scale = []
    for size in (128, 512):
        row, isolation = run_scale(size)
        scale.append(row)
        cached.append(isolation)
        print(f'SCALE {size}: {row["whole_exact"]}, {row["queries"]} queries', flush=True)
    noise = noise_benchmark(cached)
    summary = {'protocol': 'PROTOCOL_RU.md', 'main_circuits': len(evidence),
               'overall': aggregate(rows),
               'by_cohort': {cohort: aggregate([r for r in rows if r['cohort'] == cohort])
                             for cohort in sorted({r['cohort'] for r in rows})},
               'by_family': {family: aggregate([r for r in rows if r['family'] == family])
                             for family in sorted({r['family'] for r in rows})},
               'scale': [{k: v for k, v in row.items() if k not in
                          ('anonymized_true_pairs_FOR_SCORING_ONLY', 'responses_hex')} for row in scale],
               'noise': noise, 'wall_seconds': time.perf_counter() - start,
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'inference_dependencies': 'Python standard library only; CPU; no GPU'}}
    old = summary['by_cohort']['original_50']['passive_counts']
    assert (old['correct_unique'], old['gates'], old['whole_exact']) == (2459, 3509, 5)
    assert len(rows) == len(specs) * len(METHODS) and len(evidence) == 120
    for name, value in (('rows.json', rows), ('evidence.json', evidence), ('scale_evidence.json', scale),
                        ('noise.json', noise), ('summary.json', summary)):
        write_json(args.output / name, value)
    # Check final persisted data, not only the in-memory aggregates.
    assert len(json.loads((args.output / 'rows.json').read_text())) == 720
    assert len(json.loads((args.output / 'evidence.json').read_text())) == 120
    print(json.dumps({'overall': summary['overall'], 'noise': noise, 'wall_seconds': summary['wall_seconds']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

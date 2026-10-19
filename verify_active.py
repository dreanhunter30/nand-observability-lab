#!/usr/bin/env python3
"""Independent scalar checks, negative controls and persisted result audit."""
from __future__ import annotations

import collections
import itertools
import json
import math
import random
import time
from fractions import Fraction
from pathlib import Path

import active_lab as lab
import run_active as runner

HERE = Path(__file__).resolve().parent


def scalar_graph(graph, inputs, assignment):
    """No Circuit object, packed arithmetic, or generator ordering is used."""
    result = {name: (assignment >> i) & 1 for i, name in enumerate(inputs)}
    pending = dict(graph)
    while pending:
        progressed = False
        for out in list(pending):
            x, y = pending[out]
            if x in result and y in result:
                result[out] = int(not (result[x] and result[y]))
                del pending[out]
                progressed = True
        assert progressed, 'Cycle or unknown source'
    return result


def tiny_random_checks(number=240):
    isolation_columns = intervention_columns = clamp_ignored_checks = 0
    small_gates = 0
    for seed in range(number):
        rng = random.Random(2040000000 + seed)
        c = lab.base.Circuit(rng.randrange(2, 6))
        nodes = list(c.inputs)
        for _ in range(rng.randrange(1, 10)):
            nodes.append(c.nand(rng.choice(nodes), rng.choice(nodes)))
        small_gates += len(c.gates)
        mapping, truth = lab.base.anonymous(c, 2050000000 + seed)
        responses, descriptors, width, n, mismatch = lab.isolated_observations(c, mapping)
        recovered, info = lab.decode_isolated(responses, descriptors, width, mismatch)
        assert info['certified'] and recovered == truth
        all_names = set(c.inputs) | set(truth)
        for out, _, sources in descriptors:
            for position in range(width):
                desired = {name: 0 if position == 1 else 1 for name in all_names}
                if position >= 2:
                    desired[sources[position - 2]] = 0
                assignment = sum(desired[name] << i for i, name in enumerate(c.inputs))
                overrides = {name: desired[name] for name in truth if name != out}
                actual = lab.simulate_scalar(c, mapping, assignment, overrides)
                assert actual[out] == ((responses[out] >> position) & 1)
                isolation_columns += 1
        scenarios = []
        for _ in range(40):
            forced = rng.sample(sorted(all_names), rng.randrange(min(3, len(all_names)) + 1))
            scenarios.append((rng.randrange(1 << len(c.inputs)), tuple((x, rng.randrange(2)) for x in forced)))
        wave, clamped, columns = lab.simulate_batch(c, mapping, scenarios)
        for column, (assignment, overrides) in enumerate(scenarios):
            actual = lab.simulate_scalar(c, mapping, assignment, overrides)
            assert all(((wave[name] >> column) & 1) == value for name, value in actual.items())
            intervention_columns += 1
        filtered = lab.filter_active({o: {pair} for o, pair in truth.items()}, wave, clamped, columns)
        assert filtered == {o: {pair} for o, pair in truth.items()}
        for out in truth:
            if clamped.get(out, 0):
                clamp_ignored_checks += 1
        # Verify exact scalar truth tables independently of the packed inference.
        for assignment in range(1 << len(c.inputs)):
            a = scalar_graph(recovered, c.inputs, assignment)
            b = lab.simulate_scalar(c, mapping, assignment, {})
            assert a == b
    return {'circuits': number, 'gates': small_gates,
            'scalar_isolation_columns': isolation_columns,
            'scalar_intervention_columns': intervention_columns,
            'forced_target_exclusion_checks': clamp_ignored_checks}


def counterexample_checks():
    _, waves, inputs, truth, n, alternative = lab.passive.make_small_controls()[2]
    assert len(set(waves.values())) == len(waves)
    assert collections.Counter(x for p in truth.values() for x in p) == collections.Counter(
        x for p in alternative.values() for x in p)
    public = sorted(inputs | set(truth))
    c = lab.base.Circuit(0)
    c.inputs = ['A', 'B', 'E']
    c.gates = [(o, *pair) for o, pair in truth.items()]
    alt = lab.base.Circuit(0)
    alt.inputs = list(c.inputs)
    alt.gates = [(o, *pair) for o, pair in alternative.items()]
    identity = {name: name for name in public}
    normal = [(assignment, ()) for assignment in range(n)]
    actual, _, _ = lab.simulate_batch(c, identity, normal)
    changed, _, _ = lab.simulate_batch(alt, identity, normal)
    assert actual == changed == waves
    interventions = [(assignment, ((name, value),)) for assignment in range(n)
                     for name in public for value in (0, 1)]
    actual, _, _ = lab.simulate_batch(c, identity, interventions)
    changed, _, _ = lab.simulate_batch(alt, identity, interventions)
    different = 0
    for a, b in zip(actual.values(), changed.values()):
        different |= a ^ b
    assert different
    examples = []
    for case, (assignment, forced) in enumerate(interventions):
        if (different >> case) & 1:
            examples.append({'assignment': assignment, 'forced': forced,
                             'different_nodes': [o for o in actual if ((actual[o] ^ changed[o]) >> case) & 1]})
            if len(examples) == 2:
                break
    resp, desc, width, _, mis = lab.isolated_observations(c, identity)
    recovered, info = lab.decode_isolated(resp, desc, width, mis)
    assert recovered == truth and info['certified']
    return {'all_9_passive_signals_distinct': True, 'both_graphs_have_same_control_counts': True,
            'ordinary_assignments_compared': n, 'active_scenarios_compared': len(interventions),
            'distinguishing_active_scenarios': different.bit_count(), 'examples': examples,
            'isolation_recovers_true_graph': True}


def negative_checks():
    c = lab.base.Circuit(3)
    x = c.nand(c.inputs[0], c.inputs[1])
    y = c.nand(c.inputs[1], c.inputs[2])
    out = c.nand(x, y)
    mapping, truth = lab.base.anonymous(c, 17)
    target = mapping[out]
    outputs = set(truth)
    response, descriptors, width, _, mismatch = lab.isolated_observations(c, mapping)
    result = []
    for kind in ('AND', 'OR', 'XOR'):
        r, d, w, _, mis = lab.isolated_observations(c, mapping, logic_override={target: kind})
        recovered, info = lab.decode_isolated(r, d, w, mis)
        assert not info['certified'] and target not in recovered
        result.append({'control': 'wrong_gate_' + kind, 'rejected': True, **info})
    mask = (1 << width) - 1
    for sensor, corrupted in (('stuck_zero', 0), ('stuck_one', mask), ('inverted', response[target] ^ mask)):
        changed = dict(response)
        changed[target] = corrupted
        recovered, info = lab.decode_isolated(changed, descriptors, width, mismatch)
        assert not info['certified'] and target not in recovered
        result.append({'control': 'sensor_' + sensor, 'rejected': True, **info})
    r, d, w, _, mis = lab.isolated_observations(c, mapping, fault_source=mapping[x])
    recovered, info = lab.decode_isolated(r, d, w, mis)
    assert mis > 0 and not info['certified'] and recovered == {}
    result.append({'control': 'failed_actuator_with_readback', 'rejected': True,
                   'readback_mismatch_scenarios': mis, **info})
    allowed = {mapping[x]}
    try:
        lab.isolated_observations(c, mapping, allowed_internal=allowed)
    except ValueError as error:
        result.append({'control': 'insufficient_actuator_access', 'rejected': True, 'reason': str(error)})
    else:
        raise AssertionError('Isolation must reject unavailable commands')
    domains = {o: {pair} for o, pair in truth.items()}
    for plan in (lab.single_plan(c.inputs, outputs, allowed, 17), lab.pair_plan(domains, c.inputs, allowed, 17)):
        assert all(name in set(c.inputs) | allowed for _, forced in plan for name, _ in forced)
    # Self-justifying cycles must not be accepted as a complete NAND graph.
    fake_sources = ('B', 'X00')
    fake_desc = [('A', 0, fake_sources), ('B', 4, ('A', 'X00'))]
    recovered, info = lab.decode_isolated({'A': 0b1110, 'B': 0b1110}, fake_desc, 4)
    assert not info['certified'] and not recovered
    result.append({'control': 'cyclic_reconstruction', 'rejected': True, **info})
    return result


def probability_checks():
    count = 0
    for repeats in (1, 3, 5, 7, 9):
        p = Fraction(1, 10)
        exact = sum((p ** sum(bits)) * ((1 - p) ** (repeats - sum(bits)))
                    for bits in itertools.product((0, 1), repeat=repeats) if sum(bits) > repeats // 2)
        assert math.isclose(lab.majority_error_probability(repeats, float(p)), float(exact), rel_tol=1e-12)
        count += 1
    noise = json.loads((HERE / 'results/noise.json').read_text())
    for row in noise['cases']:
        if row['mode'] == 'calibrated_majority':
            r, p, q = row['repeats'], row['physical_bit_error_probability'], row['effective_logical_bit_error_probability']
            total = row['logical_queries']
            assert q * total <= 1e-6
            assert r == 1 or lab.majority_error_probability(r - 2, p) * total > 1e-6
    rng = random.Random(721)
    errors = lab.noise_mask(100000, 0.03, rng).bit_count()
    assert abs(errors - 3000) < 6 * math.sqrt(100000 * 0.03 * 0.97)
    assert lab.noise_mask(100, 0, rng) == 0
    assert lab.noise_mask(100, 1, rng) == (1 << 100) - 1
    return {'independent_exact_majority_enumerations': count,
            'all_repeat_counts_minimal_for_predeclared_bound': True,
            'bernoulli_sampler_sanity': {'trials': 100000, 'expected_errors': 3000, 'observed_errors': errors}}


def recorded_checks():
    rows = json.loads((HERE / 'results/rows.json').read_text())
    evidence = json.loads((HERE / 'results/evidence.json').read_text())
    scales = json.loads((HERE / 'results/scale_evidence.json').read_text())
    summary = json.loads((HERE / 'results/summary.json').read_text())
    assert len(evidence) == 120 and len(rows) == 720 and len(scales) == 2
    assert {(e['cohort'], e['family'], e['seed']) for e in evidence} == set(runner.case_specs())
    assert runner.aggregate(rows) == summary['overall']
    row_index = {(r['cohort'], r['family'], r['seed'], r['method']): r for r in rows}
    assert len(row_index) == 720
    checked_unique = scalar_values = 0
    for e in evidence:
        truth = {o: tuple(pair) for o, pair in e['anonymized_true_pairs_FOR_SCORING_ONLY'].items()}
        assert sum(e['observable_control_pin_counts'].values()) == 2 * len(truth)
        assert e['MOS_record_count'] == 4 * len(truth)
        assert not set(e['training_inputs']) & set(e['holdout_inputs'])
        for method, saved_domains in e['domains'].items():
            domains = {o: {tuple(p) for p in pairs} for o, pairs in saved_domains.items()}
            assert set(domains) == set(truth)
            unique = {o: next(iter(pairs)) for o, pairs in domains.items() if len(pairs) == 1}
            correct = sum(pair == truth[o] for o, pair in unique.items())
            assert correct == len(unique)
            assert all(truth[o] in pairs for o, pairs in domains.items())
            row = row_index[e['cohort'], e['family'], e['seed'], method]
            assert row['correct_unique'] == correct
            assert row['ambiguous'] == sum(len(p) > 1 for p in domains.values())
            assert row['whole_exact'] == (correct == len(truth))
            checked_unique += len(unique)
        # Check schedules from public labels and saved domains, without true edges.
        for access, saved in e['schedules'].items():
            allowed = set(saved['allowed_internal_ids'])
            single = lab.single_plan(e['input_names'], set(e['domains']['passive_counts']), allowed,
                                     saved['single']['seed'], forbidden_external=e['holdout_inputs'])
            assert runner.digest(single) == saved['single']['schedule_sha256']
            prior = {o: {tuple(p) for p in pairs} for o, pairs in e['domains']['single_' + access].items()}
            pair = lab.pair_plan(prior, e['input_names'], allowed, saved['pair']['seed'],
                                 forbidden_external=e['holdout_inputs'])
            assert runner.digest(pair) == saved['pair']['schedule_sha256']
            assert all(name in set(e['input_names']) | allowed
                       for _, commands in single + pair for name, _ in commands)
            assert all(lab.external_assignment(a, commands, e['input_names']) not in set(e['holdout_inputs'])
                       for a, commands in single + pair)
        # Every isolation response equals the analytical NAND parent test.
        names = sorted(set(e['input_names']) | set(truth))
        for out, pair in truth.items():
            sources = [name for name in names if name != out]
            expected = 2
            for i, source in enumerate(sources, 2):
                if source in pair:
                    expected |= 1 << i
            assert int(e['isolated']['responses_hex'][out], 16) == expected
        recovered = {o: tuple(pairs[0]) for o, pairs in e['domains']['pair_full_access'].items()}
        c = runner.make_case(e['family'], e['seed'])
        mapping, _ = lab.base.anonymous(c, e['seed'] + 200000)
        for assignment in e['holdout_inputs']:
            actual = lab.simulate_scalar(c, mapping, assignment, {})
            predicted = scalar_graph(recovered, e['input_names'], assignment)
            assert actual == predicted
            scalar_values += len(truth)
    for s in scales:
        c = runner.make_scale(s['gates'], s['seed'])
        mapping, _ = lab.base.anonymous(c, s['seed'] + 200000)
        recovered = {o: tuple(pair) for o, pair in s['anonymized_true_pairs_FOR_SCORING_ONLY'].items()}
        names = sorted(set(c.inputs) | set(recovered))
        desc = [(out, i * s['width'], tuple(x for x in names if x != out))
                for i, out in enumerate(sorted(recovered))]
        graph, info = lab.decode_isolated({o: int(v, 16) for o, v in s['responses_hex'].items()}, desc, s['width'])
        assert info['certified'] and graph == recovered
        for assignment in random.Random(s['seed'] + 1).sample(range(1 << len(c.inputs)), 256):
            assert scalar_graph(graph, c.inputs, assignment) == lab.simulate_scalar(c, mapping, assignment, {})
            scalar_values += len(graph)
    old = summary['by_cohort']['original_50']['passive_counts']
    assert (old['correct_unique'], old['gates'], old['whole_exact']) == (2459, 3509, 5)
    assert summary['overall']['pair_full_access']['whole_exact'] == 120
    return {'main_circuits': 120, 'scale_circuits': 2, 'result_rows': len(rows),
            'unique_pair_claims_checked': checked_unique, 'holdout_scalar_internal_values_checked': scalar_values,
            'old_baseline_exactly_reproduced': True, 'public_schedule_hashes_verified': True}


def main():
    begin = time.perf_counter()
    result = {'random_small': tiny_random_checks(), 'passive_counterexample': counterexample_checks(),
              'negative_controls': negative_checks(), 'probability_math': probability_checks(),
              'persisted_results': recorded_checks(), 'all_passed': True,
              'seconds': time.perf_counter() - begin}
    runner.write_json(HERE / 'results/verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

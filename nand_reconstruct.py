#!/usr/bin/env python3
"""Infer a NAND DAG from exact all-node observations, without hidden wiring."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import active_lab as lab

FORMAT = 'nand-observations-v1'


def names(value, field):
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{field} must be a nonempty list of distinct nonempty strings')
    return value


def binary(value):
    return type(value) in (int, bool) and value in (0, 1)


def decode_observations(data):
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ValueError(f'format must be {FORMAT}')
    allowed = {'format', 'primary_inputs', 'gate_outputs', 'experiments', 'description'}
    if set(data) - allowed:
        raise ValueError('Unexpected top-level fields; hidden wiring is not accepted')
    inputs = names(data.get('primary_inputs'), 'primary_inputs')
    outputs = names(data.get('gate_outputs'), 'gate_outputs')
    if set(inputs) & set(outputs):
        raise ValueError('primary_inputs and gate_outputs must be disjoint')
    experiments = data.get('experiments')
    if not isinstance(experiments, list) or not experiments:
        raise ValueError('experiments must be a nonempty list')
    all_names = set(inputs) | set(outputs)
    waves = {name: 0 for name in sorted(all_names)}
    clamped = {name: 0 for name in outputs}
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict) or set(experiment) - {'levels', 'forced'}:
            raise ValueError(f'experiment {index}: only levels and forced are accepted')
        levels, forced = experiment.get('levels'), experiment.get('forced', {})
        if not isinstance(levels, dict) or set(levels) != all_names:
            raise ValueError(f'experiment {index}: every declared node must have a level')
        if not isinstance(forced, dict) or not set(forced) <= all_names:
            raise ValueError(f'experiment {index}: forced contains unknown nodes')
        if any(not binary(value) for value in list(levels.values()) + list(forced.values())):
            raise ValueError(f'experiment {index}: levels and commands must be binary')
        if any(levels[name] != value for name, value in forced.items()):
            raise ValueError(f'experiment {index}: actuation/readback mismatch')
        for name, value in levels.items():
            waves[name] |= int(value) << index
        for name in forced:
            if name in clamped:
                clamped[name] |= 1 << index
    return inputs, outputs, waves, clamped, len(experiments)


def reconstruct(data):
    inputs, outputs, waves, clamped, samples = decode_observations(data)
    candidates = {out: set(itertools.combinations_with_replacement(
        sorted(set(waves) - {out}), 2)) for out in outputs}
    candidates = lab.filter_active(candidates, waves, clamped, samples)
    candidates, feasible = lab.base.prune_acyclic(candidates, set(inputs))
    unique = {out: next(iter(pairs)) for out, pairs in candidates.items() if len(pairs) == 1}
    if not feasible:
        status = 'inconsistent_with_declared_NAND_DAG_model'
    elif len(unique) == len(outputs):
        assert lab.passive.validate_graph(unique, set(inputs))
        status = 'unique_under_declared_NAND_DAG_model'
    else:
        status = 'ambiguous_under_declared_NAND_DAG_model'
    return {
        'format': 'nand-reconstruction-v1', 'status': status,
        'declared_gate_count': len(outputs), 'experiments': samples,
        'unique_pairs': {out: list(pair) for out, pair in sorted(unique.items())},
        'candidates': {out: [list(pair) for pair in sorted(pairs)] for out, pairs in sorted(candidates.items())},
        'ambiguous_gate_count': sum(len(pairs) > 1 for pairs in candidates.values()),
        'empty_gate_count': sum(not pairs for pairs in candidates.values()),
        'unforced_samples_by_gate': {out: samples - clamped[out].bit_count() for out in sorted(outputs)},
        'assumptions': ['all NAND outputs observed', 'known two-input NAND gates', 'acyclic combinational wiring',
                        'exact stable binary readings', 'hard interventions with exact readback'],
        'MOS_counts_used': False, 'hidden_wiring_accepted': False,
        'noise_guarantee': None,
    }


def reject_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON file, or - for stdin')
    parser.add_argument('--output', default='-', help='JSON result file, or - for stdout')
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.input == '-' else Path(args.input).read_text(encoding='utf-8')
        data = json.loads(text, object_pairs_hook=reject_duplicate_keys)
        result = reconstruct(data)
        encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output == '-':
            sys.stdout.write(encoded)
        else:
            Path(args.output).write_text(encoded, encoding='utf-8')
    except (OSError, ValueError, TypeError) as error:
        print(f'Input error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Active NAND reconstruction: Python 3.10+, CPU, standard library only."""
from __future__ import annotations

import collections
import itertools
import math
import random

import baseline_generalization as base
import passive_counts as passive


def make_hard(kind, seed):
    rng = random.Random(seed)
    c = base.Circuit(12)
    if kind == 'constant_blocks':
        zeros = []
        for i in range(8):
            x = c.inputs[i]
            one = c.nand(x, c.inv(x))
            zeros.append(c.inv(one))
        for _ in range(8):
            x, y = rng.sample(zeros, 2)
            c.outputs.append(c.nand(x, y))
    elif kind == 'cloned_sources':
        clones = []
        for _ in range(12):
            clones.append(c.nand(c.inputs[0], c.inputs[1]))
        for _ in range(20):
            x, y = rng.sample(clones, 2)
            c.outputs.append(c.nand(x, y))
    elif kind == 'deep_chain':
        previous = c.inputs[0]
        for _ in range(128):
            previous = c.nand(previous, rng.choice(c.inputs))
        c.outputs = [previous]
    elif kind == 'reconvergent':
        roots = []
        for _ in range(6):
            a, b, d = rng.sample(c.inputs, 3)
            left, right = c.xor(a, b), c.xor(a, d)
            roots.append(c.xor(left, right))
        for _ in range(10):
            x, y = rng.sample(roots, 2)
            c.outputs.append(c.nand(x, y))
    else:
        raise ValueError(kind)
    return c


def stimulus_sets(c, seed, train=1024, test=256):
    cases = random.Random(seed + 100000).sample(range(1 << len(c.inputs)), train + test)
    return cases[:train], cases[train:]


def simulate_batch(c, mapping, scenarios):
    """Private generator. Scenarios refer only to public sensor labels.

    scenarios = [(external assignment, ((public node, forced level), ...))].
    Output arrays are packed along experiment index; no private edges returned.
    """
    n = len(scenarios)
    if n == 0:
        return {}, {}, 0
    mask = (1 << n) - 1
    wave = {name: 0 for name in c.inputs}
    clamps, ones = collections.defaultdict(int), collections.defaultdict(int)
    input_bit = {name: i for i, name in enumerate(c.inputs)}
    for t, (assignment, overrides) in enumerate(scenarios):
        for name, value in overrides:
            if name in input_bit:
                assignment = (assignment & ~(1 << input_bit[name])) | (value << input_bit[name])
            else:
                clamps[name] |= 1 << t
                if value:
                    ones[name] |= 1 << t
        for name, bit in input_bit.items():
            if (assignment >> bit) & 1:
                wave[name] |= 1 << t
    for private_out, private_x, private_y in c.gates:
        out, x, y = mapping[private_out], mapping[private_x], mapping[private_y]
        normal = mask ^ (wave[x] & wave[y])
        wave[out] = (normal & (mask ^ clamps[out])) | ones[out]
    return wave, dict(clamps), n


def filter_active(domains, waves, clamped, n):
    if n == 0:
        return {o: set(pairs) for o, pairs in domains.items()}
    mask = (1 << n) - 1
    result = {}
    for output, pairs in domains.items():
        valid = mask ^ clamped.get(output, 0)
        result[output] = {pair for pair in pairs
                          if ((waves[output] ^ (mask ^ (waves[pair[0]] & waves[pair[1]]))) & valid) == 0}
    return result


def external_assignment(assignment, overrides, ordered_inputs):
    bits = {name: index for index, name in enumerate(ordered_inputs)}
    for name, value in overrides:
        if name in bits:
            assignment = (assignment & ~(1 << bits[name])) | (value << bits[name])
    return assignment


def choose_context(rng, inputs, commands, forbidden):
    ordered = sorted(inputs)
    for _ in range(10000):
        assignment = rng.randrange(1 << len(ordered))
        if all(external_assignment(assignment, forced, ordered) not in forbidden for forced in commands):
            return assignment
    raise ValueError('No sampled external context avoids held-out assignments')


def single_plan(inputs, outputs, allowed, seed, contexts=4, forbidden_external=()):
    rng = random.Random(seed)
    names = sorted(set(inputs) | (set(outputs) & allowed))
    forbidden = set(forbidden_external)
    result = []
    for name in names:
        for _ in range(contexts):
            commands = [((name, value),) for value in (0, 1)]
            assignment = choose_context(rng, inputs, commands, forbidden)
            result.extend((assignment, forced) for forced in commands)
    return result


def pair_plan(domains, inputs, allowed, seed, forbidden_external=()):
    rng = random.Random(seed)
    permitted = set(inputs) | allowed
    pairs = {pair for options in domains.values() if len(options) > 1 for pair in options
             if pair[0] != pair[1] and set(pair) <= permitted}
    scenarios = []
    for x, y in sorted(pairs):
        commands = [((x, a), (y, b)) for a, b in itertools.product((0, 1), repeat=2)]
        assignment = choose_context(rng, inputs, commands, set(forbidden_external))
        scenarios.extend((assignment, forced) for forced in commands)
    return scenarios


def refine(domains, waves, clamped, n, inputs, node_class, counts):
    filtered = filter_active(domains, waves, clamped, n)
    return passive.propagate_counts(filtered, inputs, node_class, counts)


def stats(domains, truth):
    unique = {o: next(iter(ps)) for o, ps in domains.items() if len(ps) == 1}
    correct = sum(pair == truth[o] for o, pair in unique.items())
    return {'gates': len(truth), 'correct_unique': correct,
            'wrong_unique': len(unique) - correct,
            'ambiguous': sum(len(ps) > 1 for ps in domains.values()),
            'no_match': sum(not ps for ps in domains.values()),
            'true_pair_in_candidates': sum(truth[o] in ps for o, ps in domains.items()),
            'candidate_pairs': sum(map(len, domains.values())),
            'whole_exact': correct == len(truth)}


def isolated_observations(c, mapping, fault_source=None, logic_override=None, allowed_internal=None):
    """Strong reference: one free gate, every other node actively controlled.

    Per target: all-high, all-low, one-low on otherwise-high background.
    Build commands and generator responses as packed arrays, avoiding O(N^3)
    explicit command lists. Returns only observations and public commands.
    """
    inputs = set(c.inputs)
    outputs = sorted(mapping[o] for o, _, _ in c.gates)
    if (allowed_internal is not None and len(outputs) > 1
            and not set(outputs) <= set(allowed_internal)):
        raise ValueError('Full isolation requires actuator access to every internal node')
    names = sorted(inputs | set(outputs))
    width = len(names) + 1
    n = width * len(outputs)
    mask = (1 << n) - 1
    off = 0
    source_zero = {name: 0 for name in names}
    free = {}
    descriptors = []
    for block, output in enumerate(outputs):
        shift = block * width
        off |= 1 << (shift + 1)
        free[output] = ((1 << width) - 1) << shift
        source_names = [name for name in names if name != output]
        for j, name in enumerate(source_names, 2):
            source_zero[name] |= 1 << (shift + j)
        descriptors.append((output, shift, tuple(source_names)))
    desired = {name: mask ^ (off | source_zero[name]) for name in names}
    wave = {name: desired[name] for name in inputs}
    readback_mismatch = 0
    for private_out, private_x, private_y in c.gates:
        out, x, y = mapping[private_out], mapping[private_x], mapping[private_y]
        kind = (logic_override or {}).get(out, 'NAND')
        if kind == 'NAND':
            normal = mask ^ (wave[x] & wave[y])
        elif kind == 'XOR':
            normal = wave[x] ^ wave[y]
        elif kind == 'AND':
            normal = wave[x] & wave[y]
        elif kind == 'OR':
            normal = wave[x] | wave[y]
        else:
            raise ValueError(kind)
        clamped = mask ^ free[out]
        actual_command = desired[out]
        if out == fault_source:
            actual_command = 0
        wave[out] = (normal & free[out]) | (actual_command & clamped)
        readback_mismatch |= (wave[out] ^ desired[out]) & clamped
    response_mask = (1 << width) - 1
    responses = {out: (wave[out] >> shift) & response_mask for out, shift, _ in descriptors}
    return responses, descriptors, width, n, readback_mismatch.bit_count()


def decode_isolated(responses, descriptors, width, readback_mismatch=0):
    # Reject malformed public observations before interpreting any parent bits.
    invalid = {'certified': False, 'reason': 'invalid_observation_shape',
               'rejected_gates': len(descriptors)}
    if not descriptors or type(width) is not int or width < 3:
        return {}, invalid
    outputs = {out for out, _, _ in descriptors}
    all_names = outputs | set(descriptors[0][2])
    if (len(outputs) != len(descriptors) or set(responses) != outputs
            or any(len(sources) != width - 2 or len(set(sources)) != len(sources)
                   or set(sources) != all_names - {out}
                   for out, _, sources in descriptors)
            or any(type(value) is not int or not 0 <= value < (1 << width)
                   for value in responses.values())):
        return {}, invalid
    if readback_mismatch:
        return {}, {'certified': False, 'reason': 'actuation_readback_mismatch', 'rejected_gates': len(descriptors)}
    result, rejected = {}, []
    for out, _, sources in descriptors:
        response = responses[out]
        # The two calibration conditions test all-high and all-low.
        if (response & 1) != 0 or ((response >> 1) & 1) != 1:
            rejected.append(out)
            continue
        parents = [name for index, name in enumerate(sources, 2) if (response >> index) & 1]
        if len(parents) == 1:
            result[out] = (parents[0], parents[0])
        elif len(parents) == 2:
            result[out] = tuple(sorted(parents))
        else:
            rejected.append(out)
    if not rejected:
        if not passive.validate_graph(result, all_names - outputs):
            return {}, {'certified': False, 'reason': 'reconstructed_graph_has_cycle',
                        'rejected_gates': len(descriptors)}
    return result, {'certified': not rejected,
                    'reason': 'consistent_with_known_NAND_model' if not rejected else 'gate_or_sensor_model_mismatch',
                    'rejected_gates': len(rejected)}


def simulate_scalar(c, mapping, assignment, overrides):
    """Independent scalar simulator for verification, not used by inference."""
    forced = dict(overrides)
    values = {name: forced.get(name, (assignment >> i) & 1) for i, name in enumerate(c.inputs)}
    for private_o, private_x, private_y in c.gates:
        o, x, y = mapping[private_o], mapping[private_x], mapping[private_y]
        values[o] = forced[o] if o in forced else 1 - (values[x] & values[y])
    return values


def majority_error_probability(repeats, p):
    if type(repeats) is not int or repeats < 1 or repeats % 2 != 1:
        raise ValueError('repeats must be a positive odd integer')
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be a finite probability')
    return sum(math.comb(repeats, k) * p ** k * (1 - p) ** (repeats - k)
               for k in range(repeats // 2 + 1, repeats + 1))


def calibrated_repeats(total_reads, p, alpha=1e-6):
    if type(total_reads) is not int or total_reads < 1:
        raise ValueError('total_reads must be a positive integer')
    if not math.isfinite(p) or not 0 <= p < 0.5:
        raise ValueError('majority calibration requires 0 <= p < 0.5')
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between 0 and 1')
    for repeats in range(1, 1002, 2):
        q = majority_error_probability(repeats, p)
        if total_reads * q <= alpha:
            return repeats, q
    raise ValueError('Could not reach requested family-wise error bound')


def noise_mask(length, p, rng):
    """Exact independent Bernoulli bit flips, using geometric skipping."""
    if type(length) is not int or length < 0 or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('length must be nonnegative and p a finite probability')
    if p <= 0:
        return 0
    if p >= 1:
        return (1 << length) - 1
    log_survival = math.log1p(-p)
    position, result = -1, 0
    while True:
        position += int(math.log1p(-rng.random()) / log_survival) + 1
        if position >= length:
            return result
        result |= 1 << position

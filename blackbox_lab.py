#!/usr/bin/env python3
"""Exhaustive finite-model identifiability, with no MOS-count information."""
from __future__ import annotations

import collections
import itertools


def relabel(graph, permutation, inputs):
    count = len(graph)
    mapping = {i: i for i in range(inputs)}
    mapping.update({inputs + old: inputs + new for old, new in enumerate(permutation)})
    result = [None] * count
    for old, (x, y) in enumerate(graph):
        result[permutation[old]] = tuple(sorted((mapping[x], mapping[y])))
    return tuple(result)


def equivalent_labels(graph, inputs):
    """Primary inputs and final output are fixed; other internal names vary."""
    count = len(graph)
    return {relabel(graph, tuple(p) + (count - 1,), inputs)
            for p in itertools.permutations(range(count - 1))}


def canonical(graph, inputs):
    return min(equivalent_labels(graph, inputs))


def output_cone_complete(graph, inputs):
    pending, seen = [inputs + len(graph) - 1], set()
    while pending:
        node = pending.pop()
        if node < inputs or node in seen:
            continue
        seen.add(node)
        pending.extend(graph[node - inputs])
    return len(seen) == len(graph)


def enumerate_ordered(inputs, gates):
    """All unordered NAND parent pairs, ties allowed, no constant sources."""
    cases = 1 << inputs
    mask = (1 << cases) - 1
    primary = [sum(((a >> bit) & 1) << a for a in range(cases)) for bit in range(inputs)]

    def extend(graph, waves):
        if len(graph) == gates:
            yield tuple(graph), waves[-1]
            return
        for x in range(len(waves)):
            for y in range(x, len(waves)):
                yield from extend(graph + [(x, y)], waves + [mask ^ (waves[x] & waves[y])])
    yield from extend([], primary)


def evaluation_order(graph, inputs):
    known = set(range(inputs))
    remaining, order = set(range(len(graph))), []
    while remaining:
        ready = sorted(i for i in remaining if set(graph[i]) <= known)
        if not ready:
            raise ValueError('Not an acyclic graph over the declared node set')
        for gate in ready:
            known.add(inputs + gate)
            remaining.remove(gate)
            order.append(gate)
    return order


def public_scenarios(inputs, gates, max_clamps):
    scenarios = [(a, ()) for a in range(1 << inputs)]
    for size in range(1, max_clamps + 1):
        for targets in itertools.combinations(range(inputs, inputs + gates), size):
            for levels in itertools.product((0, 1), repeat=size):
                forced = tuple(zip(targets, levels))
                scenarios.extend((a, forced) for a in range(1 << inputs))
    return scenarios


def compile_scenarios(inputs, gates, scenarios):
    primary = [0] * inputs
    clamp, ones = [0] * gates, [0] * gates
    for t, (assignment, forced) in enumerate(scenarios):
        for bit in range(inputs):
            primary[bit] |= ((assignment >> bit) & 1) << t
        for node, level in forced:
            clamp[node - inputs] |= 1 << t
            ones[node - inputs] |= level << t
    return primary, clamp, ones, (1 << len(scenarios)) - 1


def evaluate_packed(graph, inputs, compiled):
    primary, clamp, ones, mask = compiled
    values = list(primary) + [0] * len(graph)
    for gate in evaluation_order(graph, inputs):
        x, y = graph[gate]
        natural = mask ^ (values[x] & values[y])
        values[inputs + gate] = (natural & (mask ^ clamp[gate])) | ones[gate]
    return values


def scalar_values(graph, inputs, assignment, forced):
    """Independent scalar evaluator: no packed values or shared gate order."""
    values = {i: (assignment >> i) & 1 for i in range(inputs)}
    commands = dict(forced)
    pending = {inputs + i: pair for i, pair in enumerate(graph)}
    while pending:
        progress = False
        for output in list(pending):
            x, y = pending[output]
            if x in values and y in values:
                values[output] = commands[output] if output in commands else int(not (values[x] and values[y]))
                del pending[output]
                progress = True
        if not progress:
            raise ValueError('Cycle or unknown parent')
    return values


def stage_definitions(inputs, gates):
    outputs = list(range(inputs, inputs + gates))
    return [
        ('external_only', 0, [outputs[-1]]),
        ('one_internal_sensor', 0, [outputs[-1], outputs[0]]),
        ('all_nodes_passive', 0, outputs),
        ('all_nodes_single_interventions', 1, outputs),
        ('all_nodes_pair_interventions', 2, outputs),
    ]


def run_space(inputs, gates=4):
    assert gates == 4, 'This finite benchmark is deliberately bounded at four NAND'
    minimum = {sum(((a >> bit) & 1) << a for a in range(1 << inputs)): 0 for bit in range(inputs)}
    ordered_counts, models = {}, {}
    for size in range(1, gates + 1):
        count = connected = 0
        for graph, function in enumerate_ordered(inputs, size):
            count += 1
            minimum[function] = min(size, minimum.get(function, size))
            if output_cone_complete(graph, inputs):
                connected += 1
                if size == gates:
                    normalized = canonical(graph, inputs)
                    if normalized in models:
                        assert models[normalized] == function
                    models[normalized] = function
        ordered_counts[str(size)] = {'all': count, 'all_gates_in_output_cone': connected}
    canonical_graphs = sorted(models)
    pool, topology_id = [], []
    for identity, graph in enumerate(canonical_graphs):
        for labeled in sorted(equivalent_labels(graph, inputs)):
            pool.append(labeled)
            topology_id.append(identity)
    assert len(set(pool)) == len(pool)
    function_models = collections.defaultdict(list)
    for i, graph in enumerate(canonical_graphs):
        function_models[models[graph]].append(i)
    stages, stage_evidence = {}, {}
    compiled = {size: compile_scenarios(inputs, gates, public_scenarios(inputs, gates, size)) for size in (0, 1, 2)}
    observed_cache = {size: [evaluate_packed(graph, inputs, data) for graph in pool] for size, data in compiled.items()}
    for name, max_clamps, sensors in stage_definitions(inputs, gates):
        grouped = collections.defaultdict(list)
        for i, values in enumerate(observed_cache[max_clamps]):
            grouped[tuple(values[sensor] for sensor in sensors)].append(i)
        groups = sorted(grouped.values(), key=lambda members: members[0])
        same_topology = [len({topology_id[i] for i in members}) == 1 for members in groups]
        stages[name] = {
            'labeled_graphs': len(pool), 'observational_classes': len(groups),
            'uniquely_identified_labeled_graphs': sum(len(m) == 1 for m in groups),
            'topology_identified_labeled_graphs': sum(len(m) for m, ok in zip(groups, same_topology) if ok),
            'topologically_ambiguous_classes': sum(not ok for ok in same_topology),
            'maximum_remaining_labeled_graphs': max(map(len, groups)),
            'minimum_remaining_labeled_graphs': min(map(len, groups)),
            'queries': len(public_scenarios(inputs, gates, max_clamps)),
            'internal_sensors': len(set(sensors) - {inputs + gates - 1}),
            'maximum_internal_actuators_per_query': max_clamps,
        }
        stage_evidence[name] = {'sensors': sensors, 'maximum_clamps': max_clamps, 'groups': groups}
    target_and = 1 << ((1 << inputs) - 1)
    and_models = function_models[target_and]
    examples = [canonical_graphs[i] for i in and_models[:2]]
    functions = [{'truth_table_hex': hex(f), 'nonisomorphic_topologies': len(ids), 'minimum_NAND_count': minimum[f]}
                 for f, ids in sorted(function_models.items())]
    minimal_ids = [i for i, graph in enumerate(canonical_graphs) if minimum[models[graph]] == gates]
    minimal_ambiguous = [f for f, ids in function_models.items() if minimum[f] == gates and len(ids) > 1]
    summary = {
        'primary_inputs': inputs, 'NAND_gates_exactly': gates,
        'every_gate_in_output_ancestor_cone': True,
        'internal_names_do_not_reveal_topological_order': True,
        'NAND_pin_order_quotiented': True, 'primary_input_names_fixed': True,
        'external_assignments_exhausted': 1 << inputs, 'MOS_counts_used': False,
        'enumerated_topological_counts': ordered_counts,
        'nonisomorphic_topologies': len(canonical_graphs), 'labeled_graphs': len(pool),
        'external_function_classes': len(function_models),
        'externally_identifiable_topologies': sum(len(ids) == 1 for ids in function_models.values()),
        'minimal_four_gate_topologies': len(minimal_ids),
        'ambiguous_functions_even_at_minimum_gate_count': len(minimal_ambiguous),
        'AND_example': {'matching_nonisomorphic_topologies': len(and_models),
                        'minimum_gate_count': minimum[target_and], 'known_gate_count_is_minimum': minimum[target_and] == gates,
                        'truth_table_ascending_assignments': [(target_and >> a) & 1 for a in range(1 << inputs)],
                        'two_witness_graphs': examples},
        'functions': functions, 'stages': stages,
    }
    evidence = {'primary_inputs': inputs, 'gates': gates, 'canonical_graphs': canonical_graphs,
                'canonical_functions': [models[g] for g in canonical_graphs], 'labeled_graphs': pool,
                'topology_id_by_labeled_graph': topology_id, 'stages': stage_evidence}
    return summary, evidence

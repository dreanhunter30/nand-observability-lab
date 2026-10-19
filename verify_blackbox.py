#!/usr/bin/env python3
"""Independent scalar replay and exhaustive coverage checks for the finite benchmark."""
import collections
import itertools
import json
import time
from pathlib import Path

import blackbox_lab as lab
from run_active import write_json

HERE = Path(__file__).resolve().parent


def matrix_identity(graph, inputs):
    """Independent isomorphism key via adjacency matrices with edge multiplicity."""
    size = inputs + len(graph)
    matrix = [[0] * size for _ in range(size)]
    for gate, pair in enumerate(graph, inputs):
        for source in pair:
            matrix[source][gate] += 1
    keys = []
    for permutation in itertools.permutations(range(inputs, size - 1)):
        order = tuple(range(inputs)) + permutation + (size - 1,)
        keys.append(tuple(matrix[source][target] for target in order[inputs:] for source in order))
    return min(keys)


def independent_exhaustion(inputs):
    counts, identities = {}, {}
    mask = (1 << (1 << inputs)) - 1
    primary = [sum(((a >> i) & 1) << a for a in range(1 << inputs)) for i in range(inputs)]
    minimum = {wave: 0 for wave in primary}
    for count in range(1, 5):
        domains = [list(itertools.combinations_with_replacement(range(inputs + gate), 2)) for gate in range(count)]
        total = connected = 0
        for graph in itertools.product(*domains):
            total += 1
            values = list(primary)
            for x, y in graph:
                # Different NAND expression from the main implementation.
                values.append((~values[x] | ~values[y]) & mask)
            minimum[values[-1]] = min(count, minimum.get(values[-1], count))
            pending, reachable = [inputs + count - 1], set()
            while pending:
                node = pending.pop()
                if node >= inputs and node not in reachable:
                    reachable.add(node)
                    pending.extend(graph[node - inputs])
            if len(reachable) == count:
                connected += 1
                if count == 4:
                    identity = matrix_identity(graph, inputs)
                    if identity in identities:
                        assert identities[identity] == values[-1]
                    identities[identity] = values[-1]
        counts[str(count)] = {'all': total, 'all_gates_in_output_cone': connected}
    return counts, identities, minimum


def verify_space(summary, evidence):
    inputs, gates = evidence['primary_inputs'], evidence['gates']
    canonical_graphs = [tuple(map(tuple, graph)) for graph in evidence['canonical_graphs']]
    pool = [tuple(map(tuple, graph)) for graph in evidence['labeled_graphs']]
    counts, identities, minimum = independent_exhaustion(inputs)
    assert counts == summary['enumerated_topological_counts']
    stored = {matrix_identity(graph, inputs): function for graph, function
              in zip(canonical_graphs, evidence['canonical_functions'])}
    assert len(stored) == len(canonical_graphs) and stored == identities
    assert len(set(pool)) == len(pool) == summary['labeled_graphs']
    assert all(lab.output_cone_complete(graph, inputs) for graph in pool)
    topology_ids = evidence['topology_id_by_labeled_graph']
    for graph, identity in zip(pool, topology_ids):
        assert matrix_identity(graph, inputs) == matrix_identity(canonical_graphs[identity], inputs)
    scenarios = lab.public_scenarios(inputs, gates, 2)
    # Check the public schedule's complete coverage independently.
    expected = set()
    for bits in itertools.product((None, 0, 1), repeat=gates):
        commands = tuple((inputs + i, bit) for i, bit in enumerate(bits) if bit is not None)
        if len(commands) <= 2:
            expected.update((assignment, commands) for assignment in range(1 << inputs))
    assert set(scenarios) == expected and len(scenarios) == len(expected)
    independent_groups = {name: collections.defaultdict(list) for name in evidence['stages']}
    scalar_columns = 0
    for index, graph in enumerate(pool):
        waves = [0] * gates
        for t, (assignment, commands) in enumerate(scenarios):
            values = lab.scalar_values(graph, inputs, assignment, commands)
            for gate in range(gates):
                waves[gate] |= values[inputs + gate] << t
            scalar_columns += 1
        for name, spec in evidence['stages'].items():
            queries = summary['stages'][name]['queries']
            mask = (1 << queries) - 1
            signature = tuple(waves[node - inputs] & mask for node in spec['sensors'])
            independent_groups[name][signature].append(index)
    for name, spec in evidence['stages'].items():
        groups = sorted(independent_groups[name].values(), key=lambda group: group[0])
        assert groups == spec['groups']
        metric = summary['stages'][name]
        assert len(groups) == metric['observational_classes']
        assert sum(len(group) == 1 for group in groups) == metric['uniquely_identified_labeled_graphs']
        assert sum(len(group) for group in groups if len({topology_ids[i] for i in group}) == 1
                   ) == metric['topology_identified_labeled_graphs']
    functions = collections.Counter(identities.values())
    for row in summary['functions']:
        f = int(row['truth_table_hex'], 16)
        assert functions[f] == row['nonisomorphic_topologies']
        assert minimum[f] == row['minimum_NAND_count']
    and_function = 1 << ((1 << inputs) - 1)
    example = summary['AND_example']
    assert example['matching_nonisomorphic_topologies'] == functions[and_function]
    assert example['minimum_gate_count'] == minimum[and_function]
    a, b = [tuple(map(tuple, graph)) for graph in example['two_witness_graphs']]
    assert matrix_identity(a, inputs) != matrix_identity(b, inputs)
    for assignment in range(1 << inputs):
        left = lab.scalar_values(a, inputs, assignment, {})[inputs + gates - 1]
        right = lab.scalar_values(b, inputs, assignment, {})[inputs + gates - 1]
        assert left == right == int(assignment == (1 << inputs) - 1)
    return {'inputs': inputs, 'exhaustive_topological_circuits': sum(r['all'] for r in counts.values()),
            'nonisomorphic_four_gate_topologies': len(identities), 'labeled_graphs_checked': len(pool),
            'scalar_intervention_columns': scalar_columns, 'all_five_observation_partitions_reproduced': True,
            'independent_adjacency_matrix_isomorphism': True, 'minimum_AND_gate_count': minimum[and_function],
            'different_AND_topologies': functions[and_function]}


def main():
    started = time.perf_counter()
    summaries = json.loads((HERE / 'results/blackbox_summary.json').read_text(encoding='utf-8'))['spaces']
    evidence = json.loads((HERE / 'results/blackbox_evidence.json').read_text(encoding='utf-8'))
    checks = [verify_space(summary, data) for summary, data in zip(summaries, evidence)]
    assert len(checks) == 2
    result = {'spaces': checks, 'all_passed': True, 'seconds': time.perf_counter() - started}
    write_json(HERE / 'results/blackbox_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

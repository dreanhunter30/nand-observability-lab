#!/usr/bin/env python3
"""Audit of experiment 10. CPU, Python 3.10+, standard library only.

Run next to original_blind_transistor_reconstruction.py:
    python verify_acyclic_reconstruction.py

The new assumption is explicit: the observed NAND network is combinational
and has no feedback loops. Every NAND output is observed, with exact binary
levels and known external stimuli. This does NOT reconstruct transistor
ownership and does NOT model physical power measurements.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import random
from pathlib import Path

import original_blind_transistor_reconstruction as original

HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE_SHA256 = "717262103d46c3e3151864f4a4693fdbcd44867068c4e1904722a35057a325ae"


def acyclic_completion(candidates, primary_inputs, forced=None):
    """Return a DAG realization, or None if no DAG realization exists.

    Each output has an OR of candidate input pairs; each pair requires both
    parents to be available. Starting from primary inputs, repeatedly add
    outputs with an available pair. A successful pass explicitly constructs
    a DAG. If a pass stalls, any proposed topological order would require
    an uncomputed parent, so a DAG completion is impossible.

    No generator netlist, original gate ordering or truth labels are used.
    """
    known = set(primary_inputs)
    remaining = set(candidates)
    realization = {}
    while remaining:
        ready = {}
        for output in sorted(remaining):
            pairs = (forced[1],) if forced and output == forced[0] else candidates[output]
            for pair in sorted(pairs):
                if set(pair) <= known:
                    ready[output] = pair
                    break
        if not ready:
            return None
        realization.update(ready)
        known.update(ready)
        remaining.difference_update(ready)
    return realization


def remove_cyclic_candidates(candidates, primary_inputs):
    return {
        output: {
            pair for pair in pairs
            if acyclic_completion(candidates, primary_inputs, (output, pair)) is not None
        }
        for output, pairs in candidates.items()
    }


def all_input_waves(width):
    """Bit-packed exact truth tables, without NumPy or per-case simulation.

    Bit t stores the value for external assignment t. The repeating binary
    patterns enumerate every A, B and CIN combination, including carry-in 1.
    """
    count = 1 << (2 * width + 1)
    mask = (1 << count) - 1

    def column(bit):
        run = 1 << bit
        return (mask // ((1 << (2 * run)) - 1)) * (((1 << run) - 1) << run)

    waves = {f"A{i}": column(i) for i in range(width)}
    waves.update({f"B{i}": column(width + i) for i in range(width)})
    waves["CIN"] = column(2 * width)
    return count, mask, waves


def exhaustive_observations(width=8, seed=1042):
    nodes, gates, sums, carry = original.build_adder(width)
    count, mask, waves = all_input_waves(width)
    primary = set(waves)
    for output, x, y in gates:  # generator only
        waves[output] = mask ^ (waves[x] & waves[y])
    shuffled = [f"S{i:03d}" for i in range(len(gates))]
    random.Random(seed).shuffle(shuffled)
    rename = {n: n for n in primary}
    rename.update(zip([o for o, _, _ in gates], shuffled))
    observations = {rename[n]: signal for n, signal in waves.items()}
    truth = {
        rename[o]: tuple(sorted((rename[x], rename[y]))) for o, x, y in gates
    }
    return observations, primary, truth, count, rename, (nodes, gates, sums, carry)


def gate_score(candidates, truth):
    return original.score_wiring(candidates, truth)


def independently_check_realization(plan, rename, generator, exhaustive_waves, width=8):
    nodes, gates, sums, carry = generator
    randomizer = random.Random(20261017)
    for _ in range(256):
        a = randomizer.randrange(1 << width)
        b = randomizer.randrange(1 << width)
        cin = randomizer.randrange(2)
        expected = original.compute(nodes, gates, width, (a, b, cin))
        assignment = a | (b << width) | (cin << (2 * width))
        assert all(((exhaustive_waves[rename[n]] >> assignment) & 1) == expected[n]
                   for n in nodes)
        values = {n: expected[n] for n in nodes if not n.startswith("G")}
        for output, (x, y) in plan.items():
            values[output] = 1 - (values[x] & values[y])
        assert all(values[rename[n]] == expected[n] for n in nodes)
        result = sum(values[rename[n]] << i for i, n in enumerate(sums))
        result |= values[rename[carry]] << width
        assert result == a + b + cin
    return 256


def independent_graph_validator():
    """Cross-check feasibility against exhaustive choices on tiny graphs."""
    rng = random.Random(178)
    checked = 0
    inputs = {"A", "B"}
    outputs = [f"S{i}" for i in range(4)]
    for _ in range(12):
        candidates = {}
        for output in outputs:
            parents = sorted((inputs | set(outputs)) - {output})
            pairs = list(itertools.combinations_with_replacement(parents, 2))
            candidates[output] = set(rng.sample(pairs, 2))
        valid = []
        for chosen in itertools.product(*(sorted(candidates[o]) for o in outputs)):
            graph = dict(zip(outputs, chosen))
            # Independent DFS cycle detection, not the reachability procedure.
            seen, active = set(), set()

            def visit(node):
                if node in inputs or node in seen:
                    return True
                if node in active:
                    return False
                active.add(node)
                okay = all(visit(parent) for parent in graph[node])
                active.remove(node)
                if okay:
                    seen.add(node)
                return okay

            if all(visit(o) for o in outputs):
                valid.append(graph)
        for output in outputs:
            for pair in candidates[output]:
                brute = any(g[output] == pair for g in valid)
                closure = acyclic_completion(candidates, inputs, (output, pair)) is not None
                assert brute == closure
                checked += 1
    return checked


def main():
    source_path = HERE / "original_blind_transistor_reconstruction.py"
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if EXPECTED_SOURCE_SHA256:
        assert source_sha256 == EXPECTED_SOURCE_SHA256
    exhaustive, inputs, truth, count, rename, generator = exhaustive_observations()
    candidates = original.infer_wiring(exhaustive, inputs, count)
    pruned = remove_cyclic_candidates(candidates, inputs)
    before = gate_score(candidates, truth)
    after = gate_score(pruned, truth)
    assert before["unambiguous"] == 9 and after["correct_unambiguous"] == 72
    plan = acyclic_completion(pruned, inputs)
    assert plan is not None
    checks = independently_check_realization(plan, rename, generator, exhaustive)
    replicas = []
    for width in (2, 4, 8, 16):
        for seed in (7, 42, 99, 20261017, 608):
            nodes, gates, _, _ = original.build_adder(width)
            n = min(1024, 1 << (2 * width + 1))
            stimuli = original.make_stimuli(random.Random(seed), n, width)
            waves, probes, secret, mos_truth, _ = original.shuffled_measurements(
                nodes, gates, width, stimuli, seed + 1000
            )
            primary = {n for n in nodes if not n.startswith("G")}
            raw = original.infer_wiring(waves, primary, len(stimuli))
            fixed = remove_cyclic_candidates(raw, primary)
            row = {"width": width, "seed": seed, "samples": len(stimuli),
                   "before": gate_score(raw, secret), "after": gate_score(fixed, secret)}
            assert row["after"]["true_pair_in_candidates"] == len(gates)
            assert row["after"]["correct_unambiguous"] == len(gates)
            replicas.append(row)

    # Negative control: different physical upstream gates with identical
    # waveforms remain indistinguishable, even in a loop-free circuit.
    mask = 0b1111
    identical = {"A": 0b1010, "B": 0b1100}
    identical["S0"] = mask ^ (identical["A"] & identical["B"])
    identical["S1"] = identical["S0"]
    identical["S2"] = mask ^ (identical["S0"] & identical["A"])
    negative = original.infer_wiring(identical, {"A", "B"}, 4)
    negative = remove_cyclic_candidates(negative, {"A", "B"})
    assert ("A", "S0") in negative["S2"] and ("A", "S1") in negative["S2"]

    result = {
        "date_utc": "2026-10-17",
        "original_stage10_sha256": source_sha256,
        "observations": "Exact levels at EVERY NAND output; known inputs; anonymized output IDs",
        "new_assumption": "Combinational, loop-free NAND circuit; no gate order supplied",
        "uses_transistor_traces_for_gate_inference": False,
        "exhaustive_8bit": {"all_external_assignments": count,
                              "before": before, "after": after,
                              "rejected_candidate_pairs": sum(map(len, candidates.values())) - sum(map(len, pruned.values())),
                              "reconstructed_network_random_sum_checks": checks},
        "replicas": replicas,
        "independent_bruteforce_graph_feasibility_checks": independent_graph_validator(),
        "negative_control_equivalent_signals_remaining_pairs": sorted(negative["S2"]),
        "limits": ["No physical chip or RTL simulation", "No noisy or missing-node reconstruction",
                   "Individual transistor ownership is still not recovered",
                   "Adder family tests do not establish generality or scientific novelty"],
    }
    (HERE / "acyclic_verification_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (HERE / "recovered_72_gate_netlist.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"All {count} inputs: {before['unambiguous']}/72 -> {after['correct_unambiguous']}/72")
    print(f"Replicas: {len(replicas)}/{len(replicas)} fully recovered")
    print(f"Independent feasibility checks: {result['independent_bruteforce_graph_feasibility_checks']}")
    print("Negative control: indistinguishable copies remain ambiguous, as expected")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MOS-control histogram constraints for blind NAND graph reconstruction.

CPU. Python 3.10+. Full joint certification needs SciPy (and its NumPy dependency).
    python -m pip install scipy
    python run_fanout.py

--propagation-only runs a weaker stdlib-only conservative filter.
No transistor ownership, analog physics, RTL, or real hardware measurements.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import baseline_generalization as base
import previous_acyclic_verifier as previous

HERE = Path(__file__).resolve().parent


def generate_unowned_mos_records(truth, node_waves, n, seed):
    """Generator only. Strip all gate/port/driver fields before inference."""
    mask = (1 << n) - 1
    records = []
    for output in sorted(truth):
        for source in truth[output]:
            control = node_waves[source]
            records.extend((("NMOS", control), ("PMOS", mask ^ control)))
    random.Random(seed).shuffle(records)
    return [(f"T{i:05d}", kind, signal) for i, (kind, signal) in enumerate(records)]


def infer_observable_counts(node_waves, mos_records, n):
    """No true wiring or gate ownership is an argument to this function."""
    groups = collections.defaultdict(list)
    for name in sorted(node_waves):
        groups[node_waves[name]].append(name)
    class_for_signal = {signal: names[0] for signal, names in groups.items()}
    node_class = {name: class_for_signal[signal] for name, signal in node_waves.items()}
    n_counts, p_counts = collections.Counter(), collections.Counter()
    mask = (1 << n) - 1
    for _, kind, conduction in mos_records:
        if kind not in ("NMOS", "PMOS"):
            raise ValueError("Unknown transistor type")
        control = conduction if kind == "NMOS" else mask ^ conduction
        if control not in class_for_signal:
            raise ValueError("Control signal has no observed matching node")
        (n_counts if kind == "NMOS" else p_counts)[class_for_signal[control]] += 1
    if n_counts != p_counts:
        raise ValueError("NMOS and PMOS complementary control counts disagree")
    counts = {class_for_signal[signal]: n_counts[class_for_signal[signal]] for signal in groups}
    classes = {names[0]: names for names in groups.values()}
    return node_class, counts, classes


def consumed(pair, node_class):
    return collections.Counter(node_class[name] for name in pair)


def validate_graph(graph, inputs, counts=None, node_class=None):
    """Independent discrete validation of a solver witness."""
    if not graph:
        return False
    known = set(inputs)
    remaining = set(graph)
    while remaining:
        ready = {o for o in remaining if set(graph[o]) <= known}
        if not ready:
            return False
        known.update(ready)
        remaining.difference_update(ready)
    if counts is not None:
        use = collections.Counter()
        for pair in graph.values():
            use.update(consumed(pair, node_class))
        if any(use[k] != v for k, v in counts.items()) or any(k not in counts for k in use):
            return False
    return True


def propagate_counts(candidates, inputs, node_class, counts):
    """Sound single-counter generalized arc consistency + acyclic filtering."""
    domains = {o: set(pairs) for o, pairs in candidates.items()}
    weights = {pair: consumed(pair, node_class) for pairs in domains.values() for pair in pairs}
    iterations = 0
    while True:
        iterations += 1
        before = sum(map(len, domains.values()))
        domains, feasible = base.prune_acyclic(domains, inputs)
        if not feasible:
            return {o: set() for o in domains}, {"consistent": False, "iterations": iterations}
        outputs = sorted(domains)
        for group, target in sorted(counts.items()):
            options = [{weights[pair].get(group, 0) for pair in domains[o]} for o in outputs]
            limit_mask = (1 << (target + 1)) - 1
            prefix = [1]
            for values in options:
                bits = 0
                for value in values:
                    bits |= prefix[-1] << value
                prefix.append(bits & limit_mask)
            if not ((prefix[-1] >> target) & 1):
                return {o: set() for o in domains}, {"consistent": False, "iterations": iterations}
            suffix = [0] * (len(outputs) + 1)
            suffix[-1] = 1
            for i in range(len(outputs) - 1, -1, -1):
                bits = 0
                for value in options[i]:
                    bits |= suffix[i + 1] << value
                suffix[i] = bits & limit_mask
            for i, out in enumerate(outputs):
                if len(options[i]) == 1:
                    continue
                reversed_suffix = int(format(suffix[i + 1], f"0{target + 1}b")[::-1], 2)
                allowed = {value for value in options[i] if prefix[i] & (reversed_suffix >> value)}
                domains[out] = {pair for pair in domains[out] if weights[pair].get(group, 0) in allowed}
                if not domains[out]:
                    return {o: set() for o in domains}, {"consistent": False, "iterations": iterations}
        if sum(map(len, domains.values())) == before:
            return domains, {"consistent": True, "iterations": iterations}


def certify_joint(candidates, inputs, node_class, counts, per_call=2.0, total_seconds=20.0, max_calls=300):
    """Return every proven feasible pair, retaining any unresolved alternatives.

    Maximize the number of as-yet-unwitnessed selected pairs. Positive objective
    witnesses certify new supported pairs; a proven optimum of zero rejects all
    remaining unwitnessed pairs together. Solver timeouts never reject a pair.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    started = time.perf_counter()
    outputs = sorted(candidates)
    if any(not candidates[o] for o in outputs):
        return {o: set() for o in outputs}, {"complete": True, "infeasible": True, "calls": 0}
    if all(len(candidates[o]) == 1 for o in outputs):
        graph = {o: next(iter(candidates[o])) for o in outputs}
        okay = validate_graph(graph, inputs, counts, node_class)
        return candidates if okay else {o: set() for o in outputs}, {
            "complete": True, "infeasible": not okay, "calls": 0,
            "supported_pairs": len(outputs) if okay else 0, "unresolved_pairs": 0,
            "witness_graphs": [graph] if okay else [], "seconds": time.perf_counter() - started}
    choices = [(out, pair) for out in outputs for pair in sorted(candidates[out])]
    n_choices, n_gates = len(choices), len(outputs)
    rank = {out: n_choices + i for i, out in enumerate(outputs)}
    row_i, col_i, values, lower, upper = [], [], [], [], []

    def constraint(coefficients, lo, hi):
        row = len(lower)
        for col, value in coefficients:
            if value:
                row_i.append(row)
                col_i.append(col)
                values.append(float(value))
        lower.append(lo)
        upper.append(hi)

    by_gate = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    for i, (out, pair) in enumerate(choices):
        by_gate[out].append((i, 1))
        for cls, multiplicity in consumed(pair, node_class).items():
            by_class[cls].append((i, multiplicity))
    for out in outputs:
        constraint(by_gate[out], 1, 1)
    for cls, target in sorted(counts.items()):
        constraint(by_class[cls], target, target)
    for i, (out, pair) in enumerate(choices):
        for parent in set(pair) - inputs:
            # rank(out) - rank(parent) >= 1 when this pair is selected.
            constraint([(rank[out], 1), (rank[parent], -1), (i, -n_gates)], 1 - n_gates, np.inf)
    matrix = csc_matrix((values, (row_i, col_i)), shape=(len(lower), n_choices + n_gates))
    con = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    lo = np.r_[np.zeros(n_choices), np.ones(n_gates)]
    hi = np.r_[np.ones(n_choices), np.full(n_gates, n_gates)]
    integer = np.r_[np.ones(n_choices, dtype=int), np.zeros(n_gates, dtype=int)]
    supported = set()
    rejected = set()
    witnesses = []
    calls = 0
    status_counts = collections.Counter()
    complete = False
    infeasible = False
    stop_reason = "time_budget"
    while calls < max_calls:
        unknown = set(range(n_choices)) - supported - rejected
        if not unknown:
            complete, stop_reason = True, "every_remaining_pair_has_witness"
            break
        remaining_time = total_seconds - (time.perf_counter() - started)
        if remaining_time <= 0:
            break
        objective = np.zeros(n_choices + n_gates)
        objective[list(unknown)] = -1.0
        result = milp(objective, integrality=integer, bounds=Bounds(lo, hi), constraints=con,
                      options={"time_limit": min(per_call, remaining_time), "mip_rel_gap": 0.0, "presolve": True})
        calls += 1
        status_counts[str(result.status)] += 1
        if result.status == 2:
            if supported:
                raise AssertionError("Solver contradicted an independently validated feasible witness")
            rejected.update(unknown)
            complete, infeasible, stop_reason = True, True, "proven_infeasible"
            break
        valid = False
        if result.x is not None:
            selected = {i for i in range(n_choices) if result.x[i] > 0.5}
            graph = {choices[i][0]: choices[i][1] for i in selected}
            valid = len(selected) == n_gates and len(graph) == n_gates and validate_graph(graph, inputs, counts, node_class)
            if valid:
                newly_seen = selected & unknown
                supported.update(selected)
                if newly_seen:
                    witnesses.append(graph)
                if result.status == 0 and not newly_seen:
                    assert abs(result.fun) < 1e-6
                    rejected.update(unknown)
                    complete, stop_reason = True, "proven_zero_unwitnessed_pairs"
                    break
        if result.status == 0 and not valid:
            raise AssertionError("Optimal solver output failed discrete validation")
    if calls >= max_calls and not complete:
        stop_reason = "call_budget"
    unknown = set(range(n_choices)) - supported - rejected
    kept = {o: set() for o in outputs}
    for i in supported | unknown:
        out, pair = choices[i]
        kept[out].add(pair)
    return kept, {"complete": complete, "infeasible": infeasible, "calls": calls,
                  "supported_pairs": len(supported), "rejected_pairs": len(rejected),
                  "unresolved_pairs": len(unknown), "stop_reason": stop_reason,
                  "status_counts": dict(status_counts), "witness_graphs": witnesses,
                  "seconds": time.perf_counter() - started}


def make_small_controls():
    simple_waves = {"A": 0b1010, "B": 0b1100, "S0": 0b0011, "S1": 0b1101, "S2": 0b0111}
    simple_truth = {"S0": ("B", "B"), "S1": ("A", "S0"), "S2": ("A", "B")}
    duplicate_waves = {"A": 0b1010, "B": 0b1100, "S0": 0b0111, "S1": 0b0111, "S2": 0b1101}
    duplicate_truth = {"S0": ("A", "B"), "S1": ("A", "B"), "S2": ("A", "S0")}
    # Two different loop-free networks preserve every node level and every
    # control histogram, despite every node's truth table being different.
    inputs = {"A", "B", "E"}
    primary = {"A": 0b10101010, "B": 0b11001100, "E": 0b11110000}
    stronger_truth = {"S0": ("B", "B"), "S1": ("A", "S0"),
                      "S2": ("A", "E"), "S3": ("S2", "S2"),
                      "S4": ("A", "B"), "S5": ("S1", "S3")}
    alternative = dict(stronger_truth)
    alternative["S4"] = ("A", "S1")
    alternative["S5"] = ("B", "S3")
    waves = dict(primary)
    alt_waves = dict(primary)
    for out, (x, y) in stronger_truth.items():
        waves[out] = 255 ^ (waves[x] & waves[y])
    for out, (x, y) in alternative.items():
        alt_waves[out] = 255 ^ (alt_waves[x] & alt_waves[y])
    assert waves == alt_waves and len(set(waves.values())) == len(waves)
    a_counts = collections.Counter(x for p in stronger_truth.values() for x in p)
    b_counts = collections.Counter(x for p in alternative.values() for x in p)
    assert a_counts == b_counts
    return [("old_distinct_signal_witness", simple_waves, {"A", "B"}, simple_truth, 4, None),
            ("identical_nodes", duplicate_waves, {"A", "B"}, duplicate_truth, 4, None),
            ("distinct_nodes_equal_counts", waves, inputs, stronger_truth, 8, alternative)]


def validate_small_instances(joint=True):
    checks = []
    for index, (name, waves, inputs, truth, n, known_alternative) in enumerate(make_small_controls()):
        probes = generate_unowned_mos_records(truth, waves, n, 5600 + index)
        node_class, counts, classes = infer_observable_counts(waves, probes, n)
        local = base.infer_exact(waves, inputs, n)
        baseline, _ = base.prune_acyclic(local, inputs)
        prop, info = propagate_counts(baseline, inputs, node_class, counts)
        assert info["consistent"]
        brute_supported = {o: set() for o in baseline}
        outputs = sorted(baseline)
        feasible_graphs = 0
        for choices in itertools.product(*(sorted(baseline[o]) for o in outputs)):
            graph = dict(zip(outputs, choices))
            if validate_graph(graph, inputs, counts, node_class):
                feasible_graphs += 1
                for out, pair in graph.items():
                    brute_supported[out].add(pair)
        assert feasible_graphs
        assert all(brute_supported[o] <= prop[o] for o in outputs)
        if joint:
            full, certificate = certify_joint(prop, inputs, node_class, counts, per_call=2.0, total_seconds=20.0)
            assert certificate["complete"] and full == brute_supported
        else:
            full, certificate = prop, {"complete": False, "calls": 0}
        if name == "old_distinct_signal_witness":
            assert all(full[o] == {truth[o]} for o in truth)
        elif name == "identical_nodes":
            assert ("A", "S0") in full["S2"] and ("A", "S1") in full["S2"]
        else:
            assert validate_graph(known_alternative, inputs, counts, node_class)
            assert all(known_alternative[o] in full[o] for o in truth)
        checks.append({"name": name, "bruteforce_feasible_graphs": feasible_graphs,
                       "baseline_unique": sum(len(p) == 1 for p in baseline.values()),
                       "counts_unique": sum(len(p) == 1 for p in full.values()),
                       "joint_complete": certificate["complete"],
                       "known_different_netlist_same_levels_and_counts": known_alternative,
                       "true_netlist": truth, "observable_classes": classes,
                       "input_pin_counts_per_class": counts})
    return checks


def aggregate(rows):
    result = []
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        fields = ("observed_gates", "total_physical_gates", "correct_unique", "wrong_unique",
                  "unique_total", "ambiguous", "no_match", "true_pair_in_candidates",
                  "candidate_pairs", "heldout_unique_functional_correct",
                  "heldout_unique_functional_wrong", "whole_physical_netlist_exact")
        row = {"method": method, "circuits": len(group)}
        row.update({f: sum(x[f] for x in group) for f in fields})
        row["coverage"] = row["correct_unique"] / row["observed_gates"]
        row["seconds"] = sum(x["seconds"] for x in group)
        row["complete_joint_certificates"] = sum(x.get("joint_complete", False) for x in group)
        row["unresolved_solver_pairs"] = sum(x.get("solver_unresolved_pairs", 0) for x in group)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--holdout", type=int, default=256)
    parser.add_argument("--per-call", type=float, default=2.0)
    parser.add_argument("--per-circuit", type=float, default=20.0)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--propagation-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse complete saved cases and rerun missing cases")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("Checking independent small controls...", flush=True)
    controls = validate_small_instances(not args.propagation_only)
    print("Small controls passed", flush=True)
    rows, evidence = [], []
    completed = set()
    if args.resume:
        old_summary = json.loads((args.out / "summary.json").read_text())
        assert old_summary["sample_count"] == args.samples and old_summary["holdout_count"] == args.holdout
        assert old_summary["protocol_sha256"] == hashlib.sha256((HERE / "PROTOCOL_RU.md").read_bytes()).hexdigest()
        rows = json.loads((args.out / "rows.json").read_text())
        evidence = json.loads((args.out / "evidence.json").read_text())
        required = {"previous_acyclic", "with_counts_propagation"}
        if not args.propagation_only:
            required.add("with_counts_joint")
        methods = collections.defaultdict(set)
        for row in rows:
            methods[(row["family"], row["seed"])].add(row["method"])
        completed = {(item["family"], item["seed"]) for item in evidence
                     if required <= methods[(item["family"], item["seed"])]}
        rows = [row for row in rows if (row["family"], row["seed"]) in completed]
        evidence = [item for item in evidence if (item["family"], item["seed"]) in completed]
        print("RESUMING:", len(completed), "complete saved cases", flush=True)
    for fi, family in enumerate(base.FAMILIES):
        for replicate in range(args.per_family):
            seed = 2026101700 + fi * 100 + replicate
            if (family, seed) in completed:
                continue
            circuit_start = time.perf_counter()
            c = base.generate(family, seed)
            n_total = 1 << len(c.inputs)
            chosen = random.Random(seed + 100000).sample(range(n_total), args.samples + args.holdout)
            train, test = chosen[:args.samples], chosen[args.samples:]
            mapping, truth = base.anonymous(c, seed + 200000)
            inputs = set(c.inputs)
            waves = {mapping[k]: v for k, v in base.evaluate_packed(c, base.primary_waves(c, train), args.samples).items()}
            heldout = {mapping[k]: v for k, v in base.evaluate_packed(c, base.primary_waves(c, test), args.holdout).items()}
            probes = generate_unowned_mos_records(truth, waves, args.samples, seed + 610000)
            node_class, counts, classes = infer_observable_counts(waves, probes, args.samples)
            assert len(probes) == 4 * len(c.gates) and sum(counts.values()) == 2 * len(c.gates)
            begin = time.perf_counter()
            local = base.infer_exact(waves, inputs, args.samples)
            baseline, feasible = base.prune_acyclic(local, inputs)
            baseline_seconds = time.perf_counter() - begin
            assert feasible
            begin = time.perf_counter()
            prop, prop_info = propagate_counts(baseline, inputs, node_class, counts)
            prop_seconds = time.perf_counter() - begin
            assert prop_info["consistent"]
            for method, domains, seconds in (("previous_acyclic", baseline, baseline_seconds),
                                             ("with_counts_propagation", prop, baseline_seconds + prop_seconds)):
                stats = base.assess(domains, truth, heldout, args.holdout, len(c.gates))
                assert stats["wrong_unique"] == 0 and stats["true_pair_in_candidates"] == len(c.gates)
                rows.append({"family": family, "seed": seed, "method": method, "seconds": seconds,
                             "mos_devices": len(probes), "observable_signal_classes": len(classes), **stats})
            certificate = None
            final_domains = prop
            if not args.propagation_only:
                final_domains, certificate = certify_joint(prop, inputs, node_class, counts,
                    per_call=args.per_call, total_seconds=args.per_circuit, max_calls=args.max_calls)
                stats = base.assess(final_domains, truth, heldout, args.holdout, len(c.gates))
                assert stats["wrong_unique"] == 0 and stats["true_pair_in_candidates"] == len(c.gates)
                rows.append({"family": family, "seed": seed, "method": "with_counts_joint",
                             "seconds": baseline_seconds + prop_seconds + certificate["seconds"],
                             "mos_devices": len(probes), "observable_signal_classes": len(classes),
                             "joint_complete": certificate["complete"],
                             "solver_unresolved_pairs": certificate["unresolved_pairs"], **stats})
            # Every claimed unique pair is also checked on every external input.
            n_all, primary_all = base.exhaustive_primary(c)
            full_waves = {mapping[k]: v for k, v in base.evaluate_packed(c, primary_all, n_all).items()}
            full_mask = (1 << n_all) - 1
            for out, pairs in final_domains.items():
                if len(pairs) == 1:
                    x, y = next(iter(pairs))
                    assert full_waves[out] == (full_mask ^ (full_waves[x] & full_waves[y]))
            evidence.append({"family": family, "seed": seed, "input_names": sorted(inputs),
                             "node_class": node_class, "observable_classes": classes,
                             "observed_control_pin_counts": counts,
                             "true_pairs_for_scoring_only": truth,
                             "prior_candidates": {k: sorted(v) for k, v in baseline.items()},
                             "counts_candidates": {k: sorted(v) for k, v in final_domains.items()},
                             "propagation_info": prop_info, "joint_certificate": certificate})
            final_stats = rows[-1]
            print(f"DONE {family} {replicate + 1}/{args.per_family}: "
                  f"{sum(len(x) == 1 for x in baseline.values())}->{final_stats['correct_unique']}/{len(c.gates)} "
                  f"complete={certificate['complete'] if certificate else False} "
                  f"calls={certificate['calls'] if certificate else 0} "
                  f"{time.perf_counter() - circuit_start:.2f}s", flush=True)
            (args.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
            (args.out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    # Commit complete final tables as well as the incremental checkpoints.
    for name, content in (("rows.json", rows), ("evidence.json", evidence)):
        temporary = args.out / (name + ".tmp")
        temporary.write_text(json.dumps(content, indent=2) + "\n")
        temporary.replace(args.out / name)
    summary = {"date_utc": "2026-10-17", "circuits": len(evidence),
               "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL_RU.md").read_bytes()).hexdigest(),
               "sample_count": args.samples, "holdout_count": args.holdout,
               "solver_limits": {"per_call_seconds": args.per_call, "per_circuit_seconds": args.per_circuit,
                                 "max_calls": args.max_calls},
               "aggregates": aggregate(rows),
               "per_family": {f: aggregate([r for r in rows if r["family"] == f]) for f in base.FAMILIES},
               "controls": controls, "elapsed_seconds": time.perf_counter() - start,
               "excluded_circuits": [],
               "resumed_complete_cases": len(completed),
               "observer": "Anonymous NAND output levels plus unowned NMOS/PMOS conductance records; counts only by indistinguishable signal classes",
               "not_tested": ["noise", "partial visibility", "physical CPU", "RTL", "individual MOS ownership"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ag = summary["aggregates"]
    with (args.out / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ag[0]))
        writer.writeheader()
        writer.writerows(ag)
    print("FINISHED", summary["circuits"], "circuits in", round(summary["elapsed_seconds"], 3), "seconds", flush=True)


if __name__ == "__main__":
    main()

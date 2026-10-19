#!/usr/bin/env python3
"""NAND netlist reconstruction generalization/stress benchmark; CPU, stdlib only.

python run_generalization.py
python run_generalization.py --per-family 1 --out pilot_results

No training, no real CPU measurements, no transistor ownership inference.
Known: NAND gates, primary inputs, loop-free combinational architecture.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import original_blind_transistor_reconstruction as reference
import previous_acyclic_verifier as previous

HERE = Path(__file__).resolve().parent
FAMILIES = ("parity_mix", "mux_bank", "equality_compare", "enabled_decoder", "random_nand_dag")


class Circuit:
    def __init__(self, n_inputs):
        self.inputs = [f"X{i:02d}" for i in range(n_inputs)]
        self.gates = []
        self.outputs = []

    def nand(self, x, y):
        out = f"G{len(self.gates):04d}"
        self.gates.append((out, x, y))
        return out

    def inv(self, x):
        return self.nand(x, x)

    def and_(self, x, y):
        return self.inv(self.nand(x, y))

    def or_(self, x, y):
        return self.nand(self.inv(x), self.inv(y))

    def xor(self, x, y):
        u = self.nand(x, y)
        return self.nand(self.nand(x, u), self.nand(y, u))

    def mux(self, x, y, sel):
        return self.nand(self.nand(x, self.inv(sel)), self.nand(y, sel))


def tree_reduce(values, combine, rng):
    values = list(values)
    while len(values) > 1:
        x = values.pop(rng.randrange(len(values)))
        y = values.pop(rng.randrange(len(values)))
        values.append(combine(x, y))
    return values[0]


def generate(family, seed):
    rng = random.Random(seed)
    if family == "parity_mix":
        c = Circuit(16)
        values = list(c.inputs)
        rng.shuffle(values)
        roots = []
        for i in range(0, 16, 4):
            roots.append(tree_reduce(values[i:i + 4], c.xor, rng))
        for _ in range(rng.randrange(2, 5)):
            x, y = rng.sample(roots, 2)
            roots.append(c.xor(x, y) if rng.randrange(2) else c.and_(x, y))
        c.outputs = roots
    elif family == "mux_bank":
        c = Circuit(11)
        data, select = c.inputs[:8], c.inputs[8:]
        for _ in range(rng.randrange(2, 4)):
            bank = rng.sample(data, len(data))
            order = rng.sample(select, len(select))
            for sel in order:
                bank = [c.mux(bank[i], bank[i + 1], sel) for i in range(0, len(bank), 2)]
            c.outputs.extend(bank)
    elif family == "equality_compare":
        width = rng.randrange(6, 9)
        c = Circuit(2 * width)
        perm = rng.sample(c.inputs, len(c.inputs))
        a, b = perm[:width], perm[width:]
        eq = [c.inv(c.xor(x, y)) for x, y in zip(a, b)]
        equality = tree_reduce(eq, c.and_, rng)
        # Unsigned lexicographic comparison, no addition/subtraction.
        less = c.and_(c.inv(a[0]), b[0])
        for x, y, same in zip(a[1:], b[1:], eq[1:]):
            less = c.or_(c.and_(c.inv(x), y), c.and_(same, less))
        c.outputs = [equality, less]
    elif family == "enabled_decoder":
        c = Circuit(11)
        select = rng.sample(c.inputs[:3], 3)
        inverse = [c.inv(x) for x in select]
        for pattern, enable in zip(rng.sample(range(8), 8), rng.sample(c.inputs[3:], 8)):
            literals = [select[i] if (pattern >> i) & 1 else inverse[i] for i in range(3)]
            term = tree_reduce(literals, c.and_, rng)
            c.outputs.append(c.and_(term, enable))
    elif family == "random_nand_dag":
        c = Circuit(14)
        available = list(c.inputs)
        for _ in range(rng.randrange(56, 81)):
            # No filtering of constants or duplicate functions.
            pool = available if rng.random() < 0.6 else c.inputs + available[-12:]
            x, y = rng.choice(pool), rng.choice(pool)
            available.append(c.nand(x, y))
        c.outputs = available[-8:]
    else:
        raise ValueError(family)
    return c


def primary_waves(c, cases):
    return {
        name: sum(((case >> bit) & 1) << t for t, case in enumerate(cases))
        for bit, name in enumerate(c.inputs)
    }


def exhaustive_primary(c):
    n = 1 << len(c.inputs)
    mask = (1 << n) - 1
    columns = {}
    for bit, name in enumerate(c.inputs):
        run = 1 << bit
        columns[name] = mask // ((1 << (2 * run)) - 1) * (((1 << run) - 1) << run)
    return n, columns


def evaluate_packed(c, primary, n):
    mask = (1 << n) - 1
    values = dict(primary)
    for out, x, y in c.gates:
        values[out] = mask ^ (values[x] & values[y])
    return values


def evaluate_scalar(c, assignment):
    values = {name: (assignment >> bit) & 1 for bit, name in enumerate(c.inputs)}
    for out, x, y in c.gates:
        values[out] = 1 - (values[x] & values[y])
    return values


def anonymous(c, seed):
    names = [f"S{i:04d}" for i in range(len(c.gates))]
    random.Random(seed).shuffle(names)
    mapping = {i: i for i in c.inputs}
    mapping.update(zip([o for o, _, _ in c.gates], names))
    truth = {mapping[o]: tuple(sorted((mapping[x], mapping[y]))) for o, x, y in c.gates}
    return mapping, truth


def infer_exact(waves, inputs, n):
    return reference.infer_wiring(waves, inputs, n)


def infer_approx(waves, inputs, n, p):
    """Known, calibrated IID readout noise; not a physical-noise model."""
    q = 1 - (1 - p) ** 3
    tolerance = math.ceil(n * q + 5 * math.sqrt(n * q * (1 - q)))
    keys = sorted(waves)
    outputs = [(k, waves[k]) for k in keys if k not in inputs]
    result = {out: set() for out, _ in outputs}
    mask = (1 << n) - 1
    for ix, x in enumerate(keys):
        x_signal = waves[x]
        for y in keys[ix:]:
            prediction = mask ^ (x_signal & waves[y])
            for out, target in outputs:
                if out != x and out != y and (prediction ^ target).bit_count() <= tolerance:
                    result[out].add((x, y))
    return result, tolerance


def prune_acyclic(candidates, primary_inputs):
    """Same feasibility criterion as the prior forced-pair verifier.

    To use a particular pair for gate g, its parents must be constructible
    without g. Compute this closure once per gate, not once per pair.
    If the initial candidate graph has no completion, safely abstain.
    """
    outputs = set(candidates)
    targets, pending, waiters = [], [], collections.defaultdict(list)
    for output in sorted(outputs):
        for pair in sorted(candidates[output]):
            option = len(targets)
            targets.append(output)
            required = set(pair) - primary_inputs
            pending.append(len(required))
            for parent in required:
                waiters[parent].append(option)
    roots = [i for i, k in enumerate(pending) if k == 0]

    def closure(blocked=None):
        remaining = list(pending)
        known = set(primary_inputs)
        todo = collections.deque()

        def add(node):
            if node != blocked and node not in known:
                known.add(node)
                todo.append(node)

        for option in roots:
            add(targets[option])
        while todo:
            parent = todo.popleft()
            for option in waiters[parent]:
                remaining[option] -= 1
                if remaining[option] == 0:
                    add(targets[option])
        return known

    if not outputs <= closure():
        return {out: set() for out in candidates}, False
    filtered = {}
    for output, pairs in candidates.items():
        known = closure(output)
        filtered[output] = {pair for pair in pairs if set(pair) <= known}
    return filtered, True


def assess(candidates, truth, clean_holdout, n_holdout, n_total):
    counts = [len(c) for c in candidates.values()]
    unique = {o: next(iter(cs)) for o, cs in candidates.items() if len(cs) == 1}
    good = sum(pair == truth[o] for o, pair in unique.items())
    mask = (1 << n_holdout) - 1
    functional_good = sum(
        clean_holdout[o] == (mask ^ (clean_holdout[x] & clean_holdout[y]))
        for o, (x, y) in unique.items()
    )
    observed = set(clean_holdout)
    eligible = sum(set(truth[o]) <= observed for o in candidates)
    contains = sum(truth[o] in cs for o, cs in candidates.items())
    return {
        "observed_gates": len(candidates), "total_physical_gates": n_total,
        "actual_parents_both_visible": eligible,
        "correct_unique": good, "wrong_unique": len(unique) - good,
        "unique_total": len(unique), "ambiguous": sum(k > 1 for k in counts),
        "no_match": sum(k == 0 for k in counts), "true_pair_in_candidates": contains,
        "candidate_pairs": sum(counts), "max_candidates": max(counts, default=0),
        "coverage_of_observed": good / len(candidates) if candidates else None,
        "coverage_of_all_gates": good / n_total,
        "precision_if_unique": good / len(unique) if unique else None,
        "heldout_unique_functional_correct": functional_good,
        "heldout_unique_functional_wrong": len(unique) - functional_good,
        "whole_physical_netlist_exact": good == n_total,
    }


def add_noise(waves, inputs, n, p, seed):
    rng = random.Random(seed)
    result = {}
    errors = 0
    for name in sorted(waves):
        word = waves[name]
        if name not in inputs:
            mask = 0
            for i in range(n):
                if rng.random() < p:
                    mask |= 1 << i
            errors += mask.bit_count()
            word ^= mask
        result[name] = word
    return result, errors


def validate_implementations():
    rng = random.Random(503)
    verified = 0
    for _ in range(32):
        inputs = {"A", "B"}
        outputs = [f"S{i}" for i in range(5)]
        cand = {}
        for out in outputs:
            parents = sorted((inputs | set(outputs)) - {out})
            choices = list(itertools.combinations_with_replacement(parents, 2))
            cand[out] = set(rng.sample(choices, rng.randrange(1, 4)))
        expected = previous.remove_cyclic_candidates(cand, inputs)
        actual, _ = prune_acyclic(cand, inputs)
        assert actual == expected
        verified += sum(map(len, cand.values()))
    # Ensure exhaustive binary patterns agree with a separate scalar evaluator.
    for i, family in enumerate(FAMILIES):
        c = generate(family, 9200 + i)
        n, primary = exhaustive_primary(c)
        packed = evaluate_packed(c, primary, n)
        for _ in range(20):
            assignment = rng.randrange(n)
            scalar = evaluate_scalar(c, assignment)
            assert all(((packed[node] >> assignment) & 1) == bit for node, bit in scalar.items())
    return verified


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["method"])].append(row)
    result = []
    for (condition, method), items in sorted(groups.items()):
        fields = ("observed_gates", "total_physical_gates", "actual_parents_both_visible",
                  "correct_unique", "wrong_unique", "unique_total", "ambiguous", "no_match",
                  "true_pair_in_candidates", "candidate_pairs", "heldout_unique_functional_correct",
                  "heldout_unique_functional_wrong", "whole_physical_netlist_exact")
        entry = {"condition": condition, "method": method, "circuits": len(items)}
        for field in fields:
            entry[field] = sum(x[field] for x in items)
        entry["coverage_of_observed"] = entry["correct_unique"] / entry["observed_gates"]
        entry["coverage_of_all_gates"] = entry["correct_unique"] / entry["total_physical_gates"]
        entry["precision_if_unique"] = entry["correct_unique"] / entry["unique_total"] if entry["unique_total"] else None
        entry["seconds"] = round(sum(x["seconds"] for x in items), 4)
        entry["no_acyclic_completion_circuits"] = sum(x.get("acyclic_model_possible") is False for x in items)
        result.append(entry)
    return result


def controls(n):
    # Identical upstream signals may be interchangeable as physical wires.
    waves = {"A": 0b1010, "B": 0b1100, "S0": 0b0111, "S1": 0b0111, "S2": 0b1101}
    candidates = infer_exact(waves, {"A", "B"}, 4)
    fixed, feasible = prune_acyclic(candidates, {"A", "B"})
    assert feasible and ("A", "S0") in fixed["S2"] and ("A", "S1") in fixed["S2"]
    # Same number of samples, deliberately violated NAND truth table (XOR).
    mismatch = infer_exact({"A": 0b1010, "B": 0b1100, "S0": 0b0110}, {"A", "B"}, 4)
    _, feasible_xor = prune_acyclic(mismatch, {"A", "B"})
    assert not feasible_xor
    return {"identical_sources_remain_ambiguous": True,
            "wrong_gate_type_rejected": True,
            "note": "Time-shuffle controls are included per circuit in the benchmark."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--holdout", type=int, default=256)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--skip-approx", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protocol_sha256 = hashlib.sha256((HERE / "PROTOCOL_RU.md").read_bytes()).hexdigest()
    checks = validate_implementations()
    print("Validation passed:", checks, "forced-pair comparisons; generators independently checked", flush=True)
    rows, metadata = [], []

    def record(family, seed, condition, clean_holdout, truth, total, waves, inputs,
               n, approximate_p=None, readout_errors=None):
        infer_start = time.perf_counter()
        if approximate_p is None:
            cand, tolerance = infer_exact(waves, inputs, n), 0
            prefix = "exact"
        else:
            cand, tolerance = infer_approx(waves, inputs, n, approximate_p)
            prefix = "calibrated_approx"
        infer_time = time.perf_counter() - infer_start
        prune_start = time.perf_counter()
        fixed, feasible = prune_acyclic(cand, inputs)
        prune_time = time.perf_counter() - prune_start
        for suffix, candidates, seconds in (("local", cand, infer_time), ("acyclic", fixed, infer_time + prune_time)):
            row = {"family": family, "seed": seed, "condition": condition,
                   "method": prefix + "_" + suffix, "samples": n,
                   "allowed_mismatches": tolerance, "readout_errors": readout_errors,
                   "seconds": round(seconds, 6),
                   "acyclic_model_possible": feasible if suffix == "acyclic" else None,
                   **assess(candidates, truth, clean_holdout, args.holdout, total)}
            rows.append(row)
        # Check algorithm does not remove ground-truth options under its valid assumptions.
        if condition in ("clean_100pct", "all_external_inputs"):
            assert all(truth[o] in fixed[o] for o in truth)
            assert rows[-1]["wrong_unique"] == 0

    for fi, family in enumerate(FAMILIES):
        for replicate in range(args.per_family):
            seed = 2026101700 + fi * 100 + replicate
            c = generate(family, seed)
            domain = 1 << len(c.inputs)
            if args.samples + args.holdout > domain:
                raise ValueError("Not enough distinct external inputs")
            chosen = random.Random(seed + 100000).sample(range(domain), args.samples + args.holdout)
            train, test = chosen[:args.samples], chosen[args.samples:]
            assert not set(train) & set(test)
            mapping, truth = anonymous(c, seed + 200000)
            inputs = set(c.inputs)
            clean = {mapping[k]: v for k, v in evaluate_packed(c, primary_waves(c, train), args.samples).items()}
            heldout = {mapping[k]: v for k, v in evaluate_packed(c, primary_waves(c, test), args.holdout).items()}
            total = len(c.gates)
            record(family, seed, "clean_100pct", heldout, truth, total, clean, inputs, args.samples)
            n_all, ep = exhaustive_primary(c)
            all_waves = {mapping[k]: v for k, v in evaluate_packed(c, ep, n_all).items()}
            record(family, seed, "all_external_inputs", heldout, truth, total, all_waves, inputs, n_all)
            for fraction in (0.75, 0.50, 0.25):
                observed_outputs = set(random.Random(seed + 300000 + int(fraction * 100)).sample(sorted(truth), max(1, round(total * fraction))))
                visible = inputs | observed_outputs
                subset = {name: word for name, word in clean.items() if name in visible}
                check_subset = {name: word for name, word in heldout.items() if name in visible}
                record(family, seed, f"visible_{int(fraction * 100)}pct", check_subset, truth, total, subset, inputs, args.samples)
            for p in (0.001, 0.01, 0.05):
                noisy, errors = add_noise(clean, inputs, args.samples, p, seed + 400000 + round(p * 1000000))
                condition = f"noise_{p:g}"
                record(family, seed, condition, heldout, truth, total, noisy, inputs, args.samples, readout_errors=errors)
                if not args.skip_approx and p in (0.01, 0.05):
                    record(family, seed, condition, heldout, truth, total, noisy, inputs, args.samples, p, errors)
            # Independent timeline permutations for individual output sensors.
            rg = random.Random(seed + 500000)
            shuffled = {}
            for name, word in clean.items():
                if name in inputs:
                    shuffled[name] = word
                else:
                    permutation = rg.sample(range(args.samples), args.samples)
                    shuffled[name] = sum(((word >> source) & 1) << target for target, source in enumerate(permutation))
            record(family, seed, "shuffled_time", heldout, truth, total, shuffled, inputs, args.samples)
            distinct = len(set(all_waves.values()))
            metadata.append({"family": family, "seed": seed, "inputs": len(inputs), "gates": total,
                             "all_input_combinations": n_all, "distinct_signal_functions": distinct,
                             "total_signal_nodes": len(all_waves), "train_holdout_overlap": 0,
                             "anonymous_true_netlist": truth, "designated_outputs": [mapping[o] for o in c.outputs]})
            print(f"DONE {family} {replicate + 1}/{args.per_family}: {total} NAND, {time.perf_counter() - start:.1f}s", flush=True)
            # Checkpoint is useful for longer CPU-only runs and reproducibility.
            (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    groups = aggregate(rows)
    family_groups = {f: aggregate([r for r in rows if r["family"] == f]) for f in FAMILIES}
    try:
        import resource
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        peak_rss_kib = None
    summary = {"protocol_sha256": protocol_sha256, "circuits": len(metadata),
               "families": list(FAMILIES), "samples": args.samples, "heldout": args.holdout,
               "elapsed_seconds": round(time.perf_counter() - start, 3),
               "peak_process_rss_platform_units": peak_rss_kib,
               "fast_vs_previous_forced_pair_comparisons": checks,
               "controls": controls(args.samples), "aggregates": groups,
               "per_family": family_groups,
               "assumptions": ["NAND-only combinational circuits", "Known primary inputs",
                               "Exact internal NODE levels in clean tests; no transistor data used",
                               "IID independent state errors, known rate for approximate method",
                               "No real hardware, RTL, physical timing, or MOS ownership"],
               "excluded_circuits": []}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "circuit_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    print("FINISHED", summary["circuits"], "circuits", summary["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()

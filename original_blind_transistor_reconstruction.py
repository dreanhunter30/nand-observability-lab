#!/usr/bin/env python3
"""
Blind reconstruction of an 8-bit CMOS NAND ripple-carry adder.

No external dependencies. Python >=3.9.

The observer sees: known input stimuli, unlabeled internal output waveforms,
and shuffled PMOS/NMOS conductance traces; the NAND wiring and transistor
ownership are hidden. These are *ideal digital observations*, not CPU hardware.
"""
import argparse
import collections
import csv
import json
import random
from pathlib import Path


def build_adder(width):
    nodes = [f"A{i}" for i in range(width)] + [f"B{i}" for i in range(width)] + ["CIN"]
    gates = []
    carry = "CIN"
    sums = []

    def nand(x, y):
        out = f"G{len(gates):03d}"
        gates.append((out, x, y))
        nodes.append(out)
        return out

    for i in range(width):
        a, b = f"A{i}", f"B{i}"
        u = nand(a, b)
        v = nand(a, u)
        w = nand(b, u)
        xor = nand(v, w)
        q = nand(xor, carry)
        r = nand(xor, q)
        s = nand(carry, q)
        sums.append(nand(r, s))
        carry = nand(u, q)
    return nodes, gates, sums, carry


def compute(nodes, gates, width, inputs):
    a, b, cin = inputs
    vals = {f"A{i}": (a >> i) & 1 for i in range(width)}
    vals.update({f"B{i}": (b >> i) & 1 for i in range(width)})
    vals["CIN"] = cin
    for out, x, y in gates:
        vals[out] = 1 - (vals[x] & vals[y])
    return vals


def trace_bits(nodes, gates, width, cases):
    vectors = {n: 0 for n in nodes}
    for t, inp in enumerate(cases):
        values = compute(nodes, gates, width, inp)
        for n, value in values.items():
            if value:
                vectors[n] |= 1 << t
    return vectors


def make_stimuli(rng, n, width, exclude=None):
    limit = 1 << width
    if n > 2 * limit * limit - len(exclude or ()):
        raise ValueError("requested more unique input samples than exist")
    used = set(exclude or ())
    cases = []
    anchors = [(0, 0, 0), (limit - 1, 0, 1), (0, limit - 1, 1),
               (limit - 1, limit - 1, 1)]
    for inp in anchors:
        if len(cases) < n and inp not in used:
            used.add(inp)
            cases.append(inp)
    while len(cases) < n:
        inp = (rng.randrange(limit), rng.randrange(limit), rng.randrange(2))
        if inp not in used:
            used.add(inp)
            cases.append(inp)
    return cases


def shuffled_measurements(nodes, gates, width, cases, seed):
    """Produce renamed output sensors and unlabeled transistor conductance traces."""
    rng = random.Random(seed)
    truth = trace_bits(nodes, gates, width, cases)
    primary = {n for n in nodes if not n.startswith("G")}
    outputs = [o for o, _, _ in gates]
    names = [f"S{i:03d}" for i in range(len(outputs))]
    rng.shuffle(names)
    rename = {x: x for x in primary}
    rename.update(zip(outputs, names))
    observed = {rename[x]: truth[x] for x in nodes}
    transistors = []
    mask = (1 << len(cases)) - 1
    for gate, x, y in gates:
        for port, wire in (("X", x), ("Y", y)):
            # Every NAND has 2 PMOS + 2 NMOS; transistor gate pin on x/y.
            for kind in ("NMOS", "PMOS"):
                signal = truth[wire] if kind == "NMOS" else (truth[wire] ^ mask)
                transistors.append({"gate": rename[gate], "wire": rename[wire],
                                    "port": port, "type": kind, "signal": signal})
    rng.shuffle(transistors)
    # Strip ALL source gate/group/wire info before reconstruction.
    probes = [(f"T{i:04d}", t["type"], t["signal"]) for i, t in enumerate(transistors)]
    secret = {rename[g]: tuple(sorted((rename[x], rename[y]))) for g, x, y in gates}
    secret_trans = {f"T{i:04d}": (t["gate"], t["wire"], t["type"])
                    for i, t in enumerate(transistors)}
    return observed, probes, secret, secret_trans, rename


def infer_wiring(waves, inputs, sample_count):
    """Find every NAND(x,y) identity consistent with all observed samples."""
    mask = (1 << sample_count) - 1
    keys = sorted(waves)
    outputs = set(keys) - inputs
    reverse = collections.defaultdict(list)
    for node in outputs:
        reverse[waves[node]].append(node)
    candidates = {node: set() for node in outputs}
    for ix, x in enumerate(keys):
        for y in keys[ix:]:
            pred = (~(waves[x] & waves[y])) & mask
            for result in reverse.get(pred, []):
                if result != x and result != y:
                    candidates[result].add((x, y))
    return candidates


def score_wiring(candidates, secret):
    known = len(secret)
    unique = {k: next(iter(c)) for k, c in candidates.items() if len(c) == 1}
    correct_unique = sum(p == secret[k] for k, p in unique.items())
    true_in = sum(secret[k] in c for k, c in candidates.items())
    no_match = sum(not c for c in candidates.values())
    ambiguities = [(k, len(candidates[k])) for k in secret if len(candidates[k]) > 1]
    return {"gates": known, "true_pair_in_candidates": true_in,
            "unambiguous": len(unique), "correct_unambiguous": correct_unique,
            "ambiguous": len(ambiguities), "no_match": no_match,
            "candidate_counts": dict(collections.Counter(len(c) for c in candidates.values()))}


def transistor_controls(waves, probes, sample_count):
    """Recover possible wire that controls EACH individual MOS, not NAND ownership."""
    mask = (1 << sample_count) - 1
    matches = {}
    for tid, mos_type, conduction in probes:
        control = conduction if mos_type == "NMOS" else (conduction ^ mask)
        matches[tid] = [n for n, w in waves.items() if w == control]
    return matches


def check_sum(gates, sums, carry, nodes, width, test_cases):
    for a, b, cin in test_cases:
        values = compute(nodes, gates, width, (a, b, cin))
        result = sum((values[wire] << i) for i, wire in enumerate(sums))
        result |= values[carry] << width
        assert result == a + b + cin, (a, b, cin, result)


def export_csv(path, candidates, secret):
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["hidden_output_sensor", "candidate_pairs", "unique_pair", "true_pair_only_for_validation"])
        for out in sorted(candidates):
            c = candidates[out]
            wr.writerow([out, len(c), " & ".join(next(iter(c))) if len(c) == 1 else "",
                         " & ".join(secret[out])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="blind_reconstruction_results")
    a = parser.parse_args()
    if not 1 <= a.width <= 16 or a.samples < 8:
        parser.error("width must be 1..16 and samples >= 8")
    rng = random.Random(a.seed)
    nodes, gates, sums, carry = build_adder(a.width)
    maximum = 1 << (2 * a.width + 1)
    if a.samples >= maximum:
        parser.error("not enough distinct inputs for independent test samples")
    train = make_stimuli(rng, a.samples, a.width)
    test = make_stimuli(rng, min(256, maximum - len(train)), a.width, set(train))
    check_sum(gates, sums, carry, nodes, a.width, train + test)
    waves, probes, secret, secret_trans, rename = shuffled_measurements(
        nodes, gates, a.width, train, a.seed + 1000)
    inputs = {f"A{i}" for i in range(a.width)} | {f"B{i}" for i in range(a.width)} | {"CIN"}
    # INFERENCE BEGINS HERE: sees only waves, shuffled transistor conduction and public inputs.
    candidates = infer_wiring(waves, inputs, len(train))
    mos = transistor_controls(waves, probes, len(train))
    stats = score_wiring(candidates, secret)
    stats["train_distinct_inputs"] = len(train)
    stats["test_distinct_inputs"] = len(test)
    stats["train_test_overlap"] = len(set(train) & set(test))
    stats["transistors"] = len(probes)
    stats["transistors_unique_control_wire"] = sum(len(v) == 1 for v in mos.values())
    stats["transistors_true_control_wire_in_candidates"] = sum(
        secret_trans[k][1] in v for k, v in mos.items())
    # If a wire drives multiple identical NMOS or PMOS, ownership cannot be distinguished
    # from their conductance traces. This is a fundamental limitation, not an ML failure.
    counts = collections.Counter((t[1], secret_trans[t[0]][1]) for t in probes)
    stats["transistors_indistinguishable_by_type_and_control"] = sum(
        count for count in counts.values() if count > 1)
    # Test the UNIQUE recovered relationships on wholly unseen inputs.
    test_waves = trace_bits(nodes, gates, a.width, test)
    test_obs = {rename[n]: v for n, v in test_waves.items()}
    bitmask = (1 << len(test)) - 1
    unique = {out: next(iter(c)) for out, c in candidates.items() if len(c) == 1}
    stats["heldout_relations_correct"] = sum(
        test_obs[out] == ((~(test_obs[x] & test_obs[y])) & bitmask)
        for out, (x, y) in unique.items())
    stats["heldout_relations_total"] = len(unique)
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    export_csv(folder / "recovered_edges.csv", candidates, secret)
    with (folder / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    # Save observations only: no true physical wiring or transistor groups are disclosed.
    with (folder / "blinded_signals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "kind", "mos_type", "signal_hex"])
        for n, trace in sorted(waves.items()):
            writer.writerow([n, "node", "", hex(trace)])
        for tid, kind, trace in probes:
            writer.writerow([tid, "transistor", kind, hex(trace)])
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    print("Saved:", folder.resolve())
    print("WARNING: transistors in the same wire/type group cannot be assigned")
    print("to a particular NAND without additional physical observations.")


if __name__ == "__main__":
    main()

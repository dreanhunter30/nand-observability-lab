# Observation format

The importer accepts exact digital observations in `nand-observations-v1` JSON. It receives no reference wiring.

| Field | Required | Meaning |
|---|---|---|
| `format` | Yes | The literal `nand-observations-v1` |
| `primary_inputs` | Yes | Nonempty list of distinct known input names |
| `gate_outputs` | Yes | Nonempty list naming every NAND output |
| `experiments` | Yes | Nonempty list of observation records |
| `description` | No | Human-readable context |

Input and output names must be disjoint. Each experiment contains `levels`, a binary value for every declared node, and optionally `forced`, a mapping from a node to its commanded binary value.

Commands must agree with the observed level. When an output is forced, that sample is excluded from the NAND-equation check for that output. Its level can still constrain other gates.

`0`/`1` and JSON booleans are accepted. Floating-point values, missing sensors, duplicate JSON keys, unknown commands, and hidden-wiring fields are rejected.

```bash
python nand_reconstruct.py observations.json --output reconstructed.json
```

Use `-` for stdin or stdout.

## Output

| Status | Interpretation |
|---|---|
| `unique_under_declared_NAND_DAG_model` | Every output has one parent pair in the declared model |
| `ambiguous_under_declared_NAND_DAG_model` | Multiple candidate pairs remain for at least one output |
| `inconsistent_with_declared_NAND_DAG_model` | No complete acyclic realization fits the observations |

`unique_pairs` and `candidates` contain unordered parent pairs. `["A", "A"]` means tied inputs. `unforced_samples_by_gate` counts observations that actually test each gate's relation.

Exit code 0 means a result was produced, including an inconsistent result. Exit code 2 means an input error.

## Requirements

The importer assumes known two-input NAND gates, a combinational DAG, complete internal sensing, and exact settled readings. It does not establish those assumptions from arbitrary measurements. It does not correct noise, which can mimic a different valid circuit. The benchmark's separate majority-vote experiment does not supply a noise guarantee for this command.

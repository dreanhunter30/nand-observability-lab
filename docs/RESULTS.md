# Recorded experiments

These experiments use an ideal digital simulator. They test the stated models and candidate spaces, rather than physical devices.

## Main benchmark

The 120 circuits include 50 preceding cases, 50 new seeds of the same five families, and 20 stress cases with constant signals, cloned sources, deep chains, and reconvergent paths. There are 8,307 NAND gates.

| Method | Correct unique pairs | Complete circuits | Ambiguous gates |
|---|---:|---:|---:|
| Passive + observable MOS class counts | 5,826 | 10 / 120 | 2,481 |
| Singles, full actuator access | 8,256 | 107 / 120 | 51 |
| Then pairs, full actuator access | 8,307 | 120 / 120 | 0 |
| Singles + pairs, half actuator access | 7,631 | 44 / 120 | 676 |

All outputs remain observable, including with half the actuators. No false unique pairs occurred in these exact-data runs. The pair stage added 648 experiments after 78,952 single-stage experiments. Passive sampling used a further 1,024 external assignments per circuit.

The ordinary holdout contains 256 assignments per circuit. Actual external assignments in single/pair interventions exclude that holdout. Source pairs are scored against the hidden graph, and completed graphs are independently replayed on holdout inputs.

Evidence: [summary](../results/summary.json), [per-method rows](../results/rows.json), [candidate sets and schedules](../results/evidence.json), [independent checks](../results/verification.json).

## External identifiability

The study exhausts 22,767 topological constructions across one through four gates. The exact-four-gate spaces contain 1,296 non-isomorphic topologies and 7,578 named realizations.

| Inputs | Structural topologies | External functions | Topologies determined by full external truth table |
|---|---:|---:|---:|
| 2 | 252 | 15 | 1 / 252 |
| 3 | 1,044 | 82 | 18 / 1,044 |

Seventeen four-gate topologies implement two-input AND, whose minimum size is two gates. More strongly, six different four-gate topologies implement three-input AND, and exhaustive enumeration confirms four gates are minimal. All six agree on every external assignment.

Complete internal observation with exhaustive single-node interventions distinguishes all 7,578 named realizations in these small spaces. This remains a finite result.

Evidence: [summary](../results/blackbox_summary.json), [candidate spaces](../results/blackbox_evidence.json), [independent verification](../results/blackbox_verification.json).

## Separate isolation-noise experiment

The full-isolation reference also includes 128- and 512-gate scale cases. With the main cohort, the noise experiment covers 122 circuits, 8,947 gates, and 1,047,954 logical target readings.

| Known independent readout error | Majority repeats | Exact circuits in the protected run | Union bound per condition |
|---|---:|---:|---:|
| 1% | 15 | 122 / 122 | 6.335 × 10⁻⁷ |
| 5% | 31 | 122 / 122 | 2.334 × 10⁻⁷ |
| 10% | 49 | 122 / 122 | 5.881 × 10⁻⁷ |

The experiment samples the exact effective error channel after majority voting, without explicitly drawing every raw repeat. The bound assumes known independent readout errors, stable gate response, and perfect commands/readback. It does not cover correlated noise, the main single/pair pipeline, or the JSON importer. Single unprotected readings produced incorrect pairs.

Evidence: [noise calculations and runs](../results/noise.json), [scale cases](../results/scale_evidence.json).

## Validation

The record includes 21 API/CLI tests, 240 small random cases, nine negative controls, and independent scalar checks. A clean extraction was run without site-packages and under a different Python hash seed; deterministic evidence matched.

[Release audit](../results/release_audit.json) and [clean-extraction record](../results/clean_extraction_check.json) are historical local snapshots. Current cross-platform checks are reported by GitHub Actions.

Scientific priority and practical hardware applicability have not been established. No comparative benchmark against the related projects in the README was performed.

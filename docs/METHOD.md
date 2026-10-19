# Method

## Reconstruction target

For each observed NAND output, the tool seeks an unordered pair of immediate source nodes. A repeated source represents tied inputs. Primary input names are known, and every NAND output is observed.

Reproducing a Boolean function and identifying original wiring are different objectives. Several circuits can implement the same function.

## Candidate inference

The importer enumerates possible source pairs for each output, excluding the output itself. It retains pairs consistent with the NAND equation in every sample where that output is not forced. Acyclic-completion checks remove pairs that cannot participate in a complete DAG.

Ambiguous candidates are preserved. A unique result is conditional on the declared node set, gate type, complete sensing, and exact readings.

## Passive MOS counts

The main benchmark additionally generates anonymous ideal NMOS/PMOS conductance traces. Matching traces to passive signal classes yields aggregate control-pin counts. Indistinguishable signals share a count; hidden physical wire identities are not supplied.

The public JSON importer does not require or use these counts.

## Active observations

The main pipeline applies single-node interventions, then pair interventions selected from remaining candidates. Plans use public labels and available actuators. The hidden graph is used by simulation and scoring, not passed to the planner or importer.

A sample with a forced output cannot test that output's NAND relation. Ignoring this would incorrectly reject true connections.

## Full-isolation reference

This stronger reference releases one target output while holding every other internal output and primary input at commanded values. All-high and all-low source conditions check the expected NAND response. Each possible source is then lowered in turn against an otherwise-high background.

Held intermediate nodes suppress indirect propagation. The target rises exactly when the lowered source is an immediate parent. One detected source means tied inputs; two give the pair. The argument assumes known NAND behavior and exact independent control.

With `P` primary inputs and `G` gates, this requires `G(P + G + 1)` target readings and up to `G - 1` simultaneous internal clamps. It is stronger access than single/pair intervention access.

## Exhaustive experiments

Small-model experiments enumerate topological constructions with two or three primary inputs and one through four NAND gates. Exact-four-gate spaces require all gates to be ancestors of the external output. This is output-cone connectivity, not assumed functional irredundancy.

NAND pin order is ignored. Primary input names and the external output are fixed; other internal names are permuted. Structural topologies and graphs assigned to named measurement channels are counted separately.

Independent verification uses scalar replay, a separate Cartesian enumeration, and adjacency-matrix isomorphism keys.

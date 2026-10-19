# NAND Observability Lab

Experiments in recovering NAND wiring from digital observations, with explicit checks for ambiguity.

**Python 3.10+ · CPU only · Standard library · MIT**

## Overview

This project asks a narrow question: **which circuit connections can be determined from the signals we can observe and control?**

It includes an exact-trace reconstruction tool, synthetic intervention benchmarks, and exhaustive small-circuit experiments. The tool returns candidate connections and preserves ambiguity when the observations do not identify a unique graph.

This is a research prototype. The experiments use known two-input NAND gates and acyclic combinational circuits. Hardware acquisition and analog circuit behavior are outside its scope.

## Quick start

```bash
git clone https://github.com/dreanhunter30/nand-observability-lab.git
cd nand-observability-lab

python nand_reconstruct.py examples/passive_ambiguous.json
python nand_reconstruct.py examples/active_exact.json --output reconstructed.json
python -m unittest discover -s tests -v
```

No installation step or third-party package is required. On Windows, `py -3` can replace `python`.

The passive example leaves multiple connections possible. The active example identifies all six gates from the supplied observations. Reference wiring is stored separately for scoring and is not an input to reconstruction.

## Using the tool

Provide a JSON file containing the known primary inputs, every NAND output, and observed binary levels in each experiment. Optional `forced` entries describe commanded interventions.

```json
{
  "format": "nand-observations-v1",
  "primary_inputs": ["A"],
  "gate_outputs": ["Y"],
  "experiments": [
    {"levels": {"A": 0, "Y": 1}},
    {"levels": {"A": 1, "Y": 0}},
    {"levels": {"A": 0, "Y": 0}, "forced": {"Y": 0}}
  ]
}
```

The result reports unique pairs, remaining candidates, and whether the data are unique, ambiguous, or inconsistent **under the declared NAND DAG model**. The importer requires exact, stable readings at every declared node; it does not perform noise correction.

See [the trace format](docs/TRACE_FORMAT.md) and [the method](docs/METHOD.md) for the input contract and assumptions.

## Recorded results

The main synthetic benchmark contains 120 circuits and 8,307 NAND gates.

| Observation and control conditions | Correct unique parent pairs | Complete circuits |
|---|---:|---:|
| Passive observations + observable MOS class counts | 5,826 / 8,307 | 10 / 120 |
| Single-node interventions, full actuator access | 8,256 / 8,307 | 107 / 120 |
| Then pair interventions, full actuator access | 8,307 / 8,307 | 120 / 120 |
| Singles + pairs, half of internal actuators available | 7,631 / 8,307 | 44 / 120 |

All internal outputs remain observable in these conditions. The 120/120 result is empirical evidence for this fixed test set; it is not a universal guarantee for pair interventions. The separate full-isolation reference uses stronger simultaneous control and has a constructive sufficiency argument.

The exhaustive experiment also finds **six distinct minimal four-NAND circuits for three-input AND**. They give the same answers on all eight external assignments. External behavior alone therefore does not generally identify the physical wiring, even when the minimum gate count is known.

Detailed definitions, noise assumptions, and evidence are in [Results](docs/RESULTS.md).

## Reproduce the experiments

```bash
python -S run_active.py
python -S verify_active.py
python -S run_blackbox.py
python -S verify_blackbox.py
python -S audit_release.py
```

`-S` disables site-package initialization. The commands regenerate the corresponding files in `results/`. Running times and environment fields may differ between machines. The GitHub workflow covers Linux and Windows with Python 3.10 and 3.12; its execution status is available in the repository's **Actions** tab.

## Project guide

| Location | Contents |
|---|---|
| `nand_reconstruct.py` | Public JSON-to-candidate-graph command |
| `active_lab.py` | Intervention simulation, filtering, isolation, and noise calculations |
| `blackbox_lab.py` | Exhaustive finite-model observability experiments |
| `run_*.py`, `verify_*.py` | Benchmark runners and independent checks |
| `examples/` | Public observations and a separate scoring reference |
| `tests/` | Input-validation, API, and command-line tests |
| `results/` | Recorded results and inspectable evidence |
| `docs/` | Method, data format, and results documentation |

Earlier Russian-language reports are retained for provenance. [NOTICE.md](NOTICE.md) explains the supporting modules and historical audit records.

## Scope and related work

The implementation assumes a known NAND gate type, no feedback, complete internal observations for the importer, and exact control/readback. It does not provide device access, infer individual MOS ownership, or reconstruct commercial CPUs from external pins.

[HAL](https://github.com/emsec/hal) and [ABC](https://github.com/berkeley-abc/abc) provide related context in netlist analysis and logic synthesis/verification. No comparative performance or scientific-priority claim is made here.

Contributions that add independently sourced traces, testable timing models, or well-defined partial-observation experiments are welcome. See [Contributing](CONTRIBUTING.md).

## License

[MIT](LICENSE).

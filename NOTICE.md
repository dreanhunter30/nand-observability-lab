# Provenance

This release continues the user's transistor/NAND reconstruction experiments and the subsequent reproducibility audit in this conversation. No HAL or ABC source code is included, and neither project is a runtime dependency.

`baseline_generalization.py`, `passive_counts.py`, `previous_acyclic_verifier.py`, and `original_blind_transistor_reconstruction.py` are retained unchanged from the preceding experiment bundle as dependencies/provenance. Their old standalone entry-point descriptions refer to earlier filenames and experiments. Use the documented commands in README.md; do not interpret the optional legacy SciPy/MILP path as a dependency of this release.

`active_lab.py` adds public-input validation relative to the preceding active experiment. `run_active.py` explicitly writes UTF-8 JSON. The deterministic active results are checked against the preceding fixed totals. The black-box finite-model experiment, public trace importer, examples, unit tests, publication audit, and release documentation are new in this release.

Generated reference graphs are confined to simulators, fixture generation and evaluation records. The public reconstruction interface rejects hidden-graph fields. Local tests also disable private generators while running inference on supplied public observations.

The project is distributed under the root MIT license. Earlier reports and audit JSON files are historical pre-publication records; publication-state fields in those snapshots do not describe the current repository. The English README and documentation describe the public interface and current scope.

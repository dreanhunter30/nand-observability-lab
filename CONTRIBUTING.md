# Contributing

Small, reproducible improvements are welcome. Describe the observation model and the failure case a change addresses.

Before submitting a change, run:

```bash
python -m unittest discover -s tests -v
python -S run_active.py
python -S verify_active.py
python -S run_blackbox.py
python -S verify_blackbox.py
python -S audit_release.py
```

Keep reference wiring separate from inference inputs. Preserve ambiguous candidates rather than presenting an arbitrary choice as unique. Changes to reported results should include regenerated evidence and an explanation of model or sampling changes.

The core uses the Python standard library. Discuss additional dependencies before introducing them. Do not commit credentials or device data you are not permitted to share.

Useful directions include independently sourced traces, explicit timing models, and carefully specified partial-sensor experiments. Hardware claims should state the measurement and intervention access actually available.

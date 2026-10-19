#!/usr/bin/env python3
"""Generate public trace fixtures; private graph is only used by this generator."""
import json
from pathlib import Path

import active_lab as lab
from run_active import write_json

HERE = Path(__file__).resolve().parent


def main():
    _, _, inputs, truth, _, _ = lab.passive.make_small_controls()[2]
    c = lab.base.Circuit(0)
    c.inputs = ['A', 'B', 'E']
    c.gates = [(out, *pair) for out, pair in truth.items()]
    mapping = {name: name for name in inputs | set(truth)}
    normal = [(assignment, ()) for assignment in range(8)]
    single = [(assignment, ((node, level),)) for node in sorted(truth)
              for level in (0, 1) for assignment in range(8)]
    for name, scenarios in (('passive_ambiguous', normal), ('active_exact', normal + single)):
        records = [{'levels': lab.simulate_scalar(c, mapping, assignment, forced), 'forced': dict(forced)}
                   for assignment, forced in scenarios]
        public = {'format': 'nand-observations-v1', 'primary_inputs': c.inputs,
                  'gate_outputs': sorted(truth), 'experiments': records,
                  'description': 'Synthetic exact digital trace; no hidden wiring or MOS ownership supplied'}
        write_json(HERE / 'examples' / (name + '.json'), public)
    write_json(HERE / 'examples/private_graph_FOR_SCORING_ONLY.json',
               {'input_names': c.inputs, 'graph': truth, 'inference_must_not_read_this_file': True})
    print(json.dumps({'examples': ['passive_ambiguous.json', 'active_exact.json'], 'private_truth_separate': True}))


if __name__ == '__main__':
    main()

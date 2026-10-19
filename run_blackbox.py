#!/usr/bin/env python3
"""Exhaustive external/partial/internal observability benchmark on small NAND DAGs."""
import argparse
import json
import time
from pathlib import Path

import blackbox_lab as lab
from run_active import write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    started = time.perf_counter()
    summaries, evidence = [], []
    for inputs in (2, 3):
        summary, detail = lab.run_space(inputs)
        summaries.append(summary)
        evidence.append(detail)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    result = {'spaces': summaries, 'seconds': time.perf_counter() - started,
              'scope': 'all connected-output-cone DAGs with exactly four two-input NAND; up to 3 primary inputs',
              'claim': 'finite exhaustive identifiability within the declared model; no hardware measurements or universal pair-intervention guarantee'}
    write_json(args.output / 'blackbox_summary.json', result)
    write_json(args.output / 'blackbox_evidence.json', evidence)
    assert summaries[0]['AND_example']['matching_nonisomorphic_topologies'] == 17
    assert summaries[1]['AND_example']['known_gate_count_is_minimum']
    assert summaries[1]['AND_example']['matching_nonisomorphic_topologies'] > 1
    print('BLACKBOX_COMPLETE', result['seconds'], flush=True)


if __name__ == '__main__':
    main()

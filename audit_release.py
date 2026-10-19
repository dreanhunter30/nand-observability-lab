#!/usr/bin/env python3
"""Check saved research evidence, the public CLI, tests and release completeness."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import nand_reconstruct as tool
from run_active import write_json

HERE = Path(__file__).resolve().parent
EXPECTED = {
    'passive_counts': (5826, 10, 2481, 0),
    'single_full_access': (8256, 107, 51, 78952),
    'pair_full_access': (8307, 120, 0, 79600),
    'single_half_access': (7570, 43, 737, 45584),
    'pair_half_access': (7631, 44, 676, 46148),
    'isolated_full_access': (8307, 120, 0, 758546),
}


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def main():
    started = time.perf_counter()
    unit = subprocess.run([sys.executable, '-S', '-m', 'unittest', 'discover', '-s', str(HERE / 'tests'), '-v'],
                          cwd=HERE, text=True, capture_output=True)
    (HERE / 'final_unit_tests.log').write_text(unit.stdout + unit.stderr, encoding='utf-8')
    assert unit.returncode == 0, unit.stderr
    count = int(re.search(r'Ran (\d+) tests?', unit.stderr).group(1))
    assert count == 21
    active, active_check = read('results/summary.json'), read('results/verification.json')
    assert active_check['all_passed']
    for method, expected in EXPECTED.items():
        row = active['overall'][method]
        assert (row['correct_unique'], row['whole_exact'], row['ambiguous'], row['cumulative_queries']) == expected
        assert row['wrong_unique'] == row['no_match'] == 0 and row['gates'] == 8307
    blackbox, blackbox_check = read('results/blackbox_summary.json'), read('results/blackbox_verification.json')
    assert blackbox_check['all_passed'] and len(blackbox['spaces']) == 2
    assert [s['AND_example']['matching_nonisomorphic_topologies'] for s in blackbox['spaces']] == [17, 6]
    assert blackbox['spaces'][1]['AND_example']['minimum_gate_count'] == 4
    total_pool = sum(s['labeled_graphs'] for s in blackbox['spaces'])
    assert total_pool == 7578
    assert all(s['stages']['all_nodes_single_interventions']['uniquely_identified_labeled_graphs'] == s['labeled_graphs']
               for s in blackbox['spaces'])
    public_results = {}
    for name in ('passive_ambiguous', 'active_exact'):
        data = read('examples/' + name + '.json')
        result = tool.reconstruct(data)
        cli = subprocess.run([sys.executable, '-S', str(HERE / 'nand_reconstruct.py'),
                              str(HERE / 'examples' / (name + '.json'))],
                             cwd=HERE.parent, text=True, capture_output=True)
        assert cli.returncode == 0 and json.loads(cli.stdout) == result
        public_results[name] = {'status': result['status'], 'unique_gates': len(result['unique_pairs']),
                                'ambiguous_gates': result['ambiguous_gate_count']}
        write_json(HERE / 'results' / ('public_' + ('active' if name == 'active_exact' else 'passive') + '_reconstruction.json'), result)
    assert public_results['active_exact']['unique_gates'] == 6
    assert public_results['passive_ambiguous']['ambiguous_gates'] == 4
    required = ['README.md', 'README_RU.md', 'LICENSE', 'NOTICE.md', 'PROTOCOL_RU.md', 'FINAL_PROTOCOL_RU.md',
                'nand_reconstruct.py', 'make_publication_release.py', '.github/workflows/tests.yml']
    assert all((HERE / path).is_file() for path in required)
    assert 'MIT License' in (HERE / 'LICENSE').read_text(encoding='utf-8')
    syntax_files = sorted(HERE.glob('*.py')) + sorted((HERE / 'tests').glob('*.py'))
    for path in syntax_files:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    credential_patterns = [re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
                           re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),
                           re.compile(r'AKIA[A-Z0-9]{16}'),
                           re.compile(r'sk-proj-[A-Za-z0-9_-]{20,}')]
    scanned = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.md', '.json', '.yml'):
            text = path.read_text(encoding='utf-8')
            assert not any(pattern.search(text) for pattern in credential_patterns), f'Credential-like string in {path.name}'
            scanned.append(str(path.relative_to(HERE)))
    result = {
        'all_passed': True, 'unit_tests_passed': count, 'active_fixed_totals_reproduced': True,
        'active_circuits': 120, 'active_gates': 8307, 'active_wrong_unique_pairs': 0,
        'small_random_scalar_cases': active_check['random_small']['circuits'],
        'negative_controls_passed': len(active_check['negative_controls']),
        'exhaustive_topological_circuits_1_to_4_gates': sum(r['exhaustive_topological_circuits'] for r in blackbox_check['spaces']),
        'exhaustive_nonisomorphic_four_gate_topologies': sum(r['nonisomorphic_four_gate_topologies'] for r in blackbox_check['spaces']),
        'exhaustive_labeled_four_gate_graphs': total_pool,
        'independent_scalar_blackbox_columns': sum(r['scalar_intervention_columns'] for r in blackbox_check['spaces']),
        'public_trace_cli': public_results, 'syntax_files_checked': len(syntax_files),
        'credential_pattern_scan': {'files': len(scanned), 'matches': 0,
                                    'scope': 'limited documented token patterns; not a comprehensive security audit'},
        'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in syntax_files},
        'local_environment': {'python': sys.version, 'platform': platform.platform(), 'site_packages_disabled': sys.flags.no_site == 1},
        'github_ci': 'configuration prepared for Linux/Windows and Python 3.10/3.12; remote workflow not run',
        'publication_assessment': {'research_prototype': 'ready for a candidate v0.1.0 publication',
                                   'production_hardware_tool': 'not established', 'scientific_novelty': 'not established',
                                   'hardware_measurements': False, 'external_repository_published': False},
        'seconds': time.perf_counter() - started,
    }
    write_json(HERE / 'results/release_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

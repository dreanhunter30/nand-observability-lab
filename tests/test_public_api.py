import copy
import io
import json
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import active_lab as lab
import nand_reconstruct as tool


def fixture(name):
    return json.loads((ROOT / 'examples' / name).read_text(encoding='utf-8'))


class PublicObservationTests(unittest.TestCase):
    def test_active_fixture_exact_without_MOS_or_truth(self):
        data = fixture('active_exact.json')
        result = tool.reconstruct(data)
        truth = fixture('private_graph_FOR_SCORING_ONLY.json')['graph']
        self.assertEqual(result['status'], 'unique_under_declared_NAND_DAG_model')
        self.assertEqual(result['unique_pairs'], truth)
        self.assertFalse(result['MOS_counts_used'])

    def test_passive_fixture_preserves_ambiguity(self):
        result = tool.reconstruct(fixture('passive_ambiguous.json'))
        self.assertEqual(result['status'], 'ambiguous_under_declared_NAND_DAG_model')
        self.assertGreater(result['ambiguous_gate_count'], 0)
        self.assertIn(['A', 'B'], result['candidates']['S4'])
        self.assertIn(['A', 'S1'], result['candidates']['S4'])

    def test_inference_runs_with_private_generators_disabled(self):
        data = fixture('active_exact.json')
        with patch.object(lab.base, 'generate', side_effect=AssertionError('private generator called')), \
             patch.object(lab.passive, 'make_small_controls', side_effect=AssertionError('private fixture called')), \
             patch.object(lab.passive, 'generate_unowned_mos_records', side_effect=AssertionError('private MOS generator called')):
            self.assertEqual(tool.reconstruct(data)['status'], 'unique_under_declared_NAND_DAG_model')

    def test_forced_output_not_tested_against_NAND_relation(self):
        data = {'format': tool.FORMAT, 'primary_inputs': ['A'], 'gate_outputs': ['Y'], 'experiments': [
            {'levels': {'A': 0, 'Y': 1}}, {'levels': {'A': 1, 'Y': 0}},
            {'levels': {'A': 0, 'Y': 0}, 'forced': {'Y': 0}}]}
        self.assertEqual(tool.reconstruct(data)['unique_pairs'], {'Y': ['A', 'A']})
        data['experiments'][-1].pop('forced')
        self.assertEqual(tool.reconstruct(data)['status'], 'inconsistent_with_declared_NAND_DAG_model')

    def test_hidden_wiring_field_rejected(self):
        data = fixture('active_exact.json')
        data['true_graph'] = {}
        with self.assertRaisesRegex(ValueError, 'hidden wiring'):
            tool.reconstruct(data)

    def test_missing_sensor_rejected(self):
        data = fixture('active_exact.json')
        del data['experiments'][0]['levels']['S0']
        with self.assertRaises(ValueError):
            tool.reconstruct(data)

    def test_readback_mismatch_rejected(self):
        data = fixture('active_exact.json')
        data['experiments'][0]['forced'] = {'S0': 1 - data['experiments'][0]['levels']['S0']}
        with self.assertRaisesRegex(ValueError, 'readback mismatch'):
            tool.reconstruct(data)

    def test_nonbinary_rejected(self):
        for bad in (2, -1, '1', 1.0, None, float('nan')):
            with self.subTest(bad=bad):
                data = fixture('active_exact.json')
                data['experiments'][0]['levels']['S0'] = bad
                with self.assertRaises(ValueError):
                    tool.reconstruct(data)

    def test_unknown_forced_node_rejected(self):
        data = fixture('active_exact.json')
        data['experiments'][0]['forced'] = {'not_a_node': 1}
        with self.assertRaises(ValueError):
            tool.reconstruct(data)

    def test_duplicate_node_or_overlap_rejected(self):
        for field, value in (('primary_inputs', ['A', 'A']), ('gate_outputs', ['A'])):
            with self.subTest(field=field):
                data = fixture('active_exact.json')
                data[field] = value
                with self.assertRaises(ValueError):
                    tool.reconstruct(data)

    def test_empty_experiments_rejected(self):
        data = fixture('active_exact.json')
        data['experiments'] = []
        with self.assertRaises(ValueError):
            tool.reconstruct(data)

    def test_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            tool.reconstruct({'format': 'other'})

    def test_duplicate_JSON_keys_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"A":0,"A":1}', object_pairs_hook=tool.reject_duplicate_keys)

    def test_record_order_does_not_change_candidates(self):
        data = fixture('active_exact.json')
        baseline = tool.reconstruct(data)
        random.Random(17).shuffle(data['experiments'])
        shuffled = tool.reconstruct(data)
        self.assertEqual(baseline, shuffled)

    def test_anonymous_channel_permutation_preserves_result(self):
        data = fixture('active_exact.json')
        channels = list(data['gate_outputs'])
        random.Random(18).shuffle(channels)
        rename = dict(zip(data['gate_outputs'], channels))
        rename.update({name: name for name in data['primary_inputs']})
        data['gate_outputs'] = [rename[name] for name in data['gate_outputs']]
        for record in data['experiments']:
            for field in ('levels', 'forced'):
                record[field] = {rename[name]: value for name, value in record[field].items()}
        result = tool.reconstruct(data)
        truth = fixture('private_graph_FOR_SCORING_ONLY.json')['graph']
        expected = {rename[o]: sorted(rename[x] for x in pair) for o, pair in truth.items()}
        self.assertEqual(result['unique_pairs'], expected)

    def test_primary_input_order_does_not_change_named_inference(self):
        data = fixture('active_exact.json')
        expected = tool.reconstruct(data)
        data['primary_inputs'].reverse()
        self.assertEqual(tool.reconstruct(data), expected)

    def test_cli_stdin_and_error_status(self):
        good = subprocess.run([sys.executable, '-S', str(ROOT / 'nand_reconstruct.py'), '-'],
                              input=json.dumps(fixture('active_exact.json')), text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'], 'unique_under_declared_NAND_DAG_model')
        bad = subprocess.run([sys.executable, '-S', str(ROOT / 'nand_reconstruct.py'), '-'],
                             input='{"format":', text=True, capture_output=True)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, '')


class IsolationAndProbabilityTests(unittest.TestCase):
    def test_invalid_isolation_shapes_rejected(self):
        for responses, descriptors, width in (({}, [], 0), ({'Y': -1}, [('Y', 0, ('A',))], 3),
                                               ({'Y': 9}, [('Y', 0, ('A',))], 3),
                                               ({'Y': 2}, [('Y', 0, ('A', 'A'))], 4)):
            with self.subTest(width=width, descriptors=descriptors):
                graph, info = lab.decode_isolated(responses, descriptors, width)
                self.assertFalse(info['certified'])
                self.assertEqual(graph, {})

    def test_probability_validation(self):
        for repeats, p in ((0, .1), (2, .1), (-1, .1), (3, -1), (3, 2), (3, math.nan)):
            with self.subTest(repeats=repeats, p=p):
                with self.assertRaises(ValueError):
                    lab.majority_error_probability(repeats, p)
        for total, p, alpha in ((0, .1, 1e-6), (10, .5, 1e-6), (10, .1, 0), (10, math.nan, 1e-6)):
            with self.subTest(total=total, p=p, alpha=alpha):
                with self.assertRaises(ValueError):
                    lab.calibrated_repeats(total, p, alpha)

    def test_tied_pair_isolation_and_access(self):
        c = lab.base.Circuit(1)
        c.nand(c.inputs[0], c.inputs[0])
        mapping, truth = lab.base.anonymous(c, 1)
        response, descriptors, width, _, mismatch = lab.isolated_observations(c, mapping, allowed_internal=set())
        recovered, info = lab.decode_isolated(response, descriptors, width, mismatch)
        self.assertTrue(info['certified'])
        self.assertEqual(recovered, truth)

    def test_noise_sampler_rejects_invalid_channel(self):
        for length, p in ((-1, .1), (1, -.1), (1, math.nan)):
            with self.assertRaises(ValueError):
                lab.noise_mask(length, p, random.Random(1))


if __name__ == '__main__':
    unittest.main()

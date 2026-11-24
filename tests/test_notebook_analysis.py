"""Exercise numerical notebook functions without downloading models or datasets."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/module2_facial_motion_analysis.ipynb'


def notebook_functions():
    namespace = {'np': np}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        if any(line.lstrip().startswith(('%', '!')) for line in code.splitlines()):
            continue
        tree = ast.parse(code)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


class MotionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = notebook_functions()

    def align(self, gt, pred, gt_fps=60, pred_fps=30, confirmed=True):
        return self.functions['align_mesh_sequences'](gt, pred, gt_fps, pred_fps, confirmed)

    def test_equal_times_at_different_rates(self):
        gt = np.broadcast_to((np.arange(7) / 60)[:, None, None], (7, 5023, 3)).copy()
        pred = np.broadcast_to((np.arange(4) / 30)[:, None, None], (4, 5023, 3)).copy()
        aligned, prediction, times = self.align(gt.reshape(7, -1), pred.reshape(4, -1))
        np.testing.assert_allclose(aligned, prediction, atol=1e-12)
        np.testing.assert_allclose(times, [0, 1/30, 2/30, 3/30])

    def test_interpolates_without_stretching_time(self):
        gt = np.broadcast_to((np.arange(7) / 30)[:, None, None], (7, 5023, 3)).copy()
        pred = np.broadcast_to((np.arange(5) / 24)[:, None, None], (5, 5023, 3)).copy()
        aligned, prediction, _ = self.align(gt, pred, 30, 24)
        np.testing.assert_allclose(aligned, prediction, atol=1e-12)

    def test_unverified_timing_is_rejected(self):
        mesh = np.zeros((3, 5023, 3))
        for fps in [None, 0, -1, np.nan]:
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.align(mesh, mesh, fps, 30)
        with self.assertRaises(ValueError):
            self.align(mesh, mesh, confirmed=False)

    def test_duration_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duration mismatch'):
            self.align(np.zeros((278, 5023, 3)), np.zeros((138, 5023, 3)), 30, 30)

    def test_nonfinite_or_wrong_topology_is_rejected(self):
        for bad in [np.zeros((3, 478, 3)), np.full((3, 5023, 3), np.nan)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.align(bad, np.zeros((3, 5023, 3)))

    def test_normalization_removes_translation_and_scale(self):
        base = np.random.default_rng(42).normal(size=(478, 3))
        lm = np.stack([base, 2 * base + np.array([3, 4, 5])])
        normalized = self.functions['normalize_landmarks'](lm)
        np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-12)

    def test_degenerate_and_nonfinite_landmarks_are_rejected(self):
        for lm in [np.zeros((2, 478, 3)), np.full((2, 478, 3), np.nan), np.zeros((2, 5023, 3))]:
            with self.subTest(shape=lm.shape), self.assertRaises(ValueError):
                self.functions['normalize_landmarks'](lm)

    def test_flame_indices_are_not_silently_filtered(self):
        lm = np.zeros((2, 478, 3))
        for indices in [[1, 501], [-1], [], [1.5]]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.functions['motion_timeseries'](lm, indices)

    def test_displacement_matches_known_motion(self):
        lm = np.zeros((3, 478, 3))
        lm[1, 0] = [3, 4, 0]
        lm[2, 0] = [0, 0, 12]
        actual = self.functions['motion_timeseries'](lm, [0, 1])
        np.testing.assert_allclose(actual, [0, 2.5, 6])


if __name__ == '__main__':
    unittest.main()

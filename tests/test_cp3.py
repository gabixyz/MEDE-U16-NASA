import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reduce_cp3 as cp3


class CP3Tests(unittest.TestCase):
    def test_nonoverlapping_never_crosses_engine(self):
        frame = pd.DataFrame({
            "engine_id": [1] * 11 + [2] * 11,
            "window_start": list(range(1, 12)) * 2,
            "window_end": list(range(10, 21)) * 2,
        })
        selected = cp3.nonoverlapping(frame, 10)
        for _, group in selected.groupby("engine_id"):
            self.assertTrue((group.window_start.to_numpy()[1:] >
                             group.window_end.to_numpy()[:-1]).all())
        self.assertEqual(len(selected), 4)

    def test_empty_nonoverlap_raises(self):
        with self.assertRaisesRegex(ValueError, "No complete"):
            cp3.nonoverlapping(pd.DataFrame({"window_end": [11]}), 10)

    def test_mse_common_sensor_target(self):
        x = np.zeros((3, 3))
        reconstructed = np.array([[1., 1., 100.]] * 3)
        self.assertEqual(cp3.reconstruction_mse(x, reconstructed, [0, 1]), 1.)
        self.assertGreater(cp3.reconstruction_mse(x, reconstructed), 1000)

    def test_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            cp3.reconstruction_mse(np.zeros((2, 3)), np.zeros((2, 2)))

    def test_bottleneck_is_actual_decoder_input(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 5))
        model = MLPRegressor(hidden_layer_sizes=(32, 2, 32), activation="tanh",
                             max_iter=500, tol=.02, random_state=42).fit(x, x)
        z = cp3.encode_autoencoder(model, x)
        self.assertEqual(z.shape, (80, 2))
        np.testing.assert_allclose(cp3.decode_autoencoder(model, z), model.predict(x))

    def test_neighbor_overlap_rotation_invariant(self):
        z = np.random.default_rng(42).normal(size=(80, 2))
        self.assertAlmostEqual(cp3.neighbor_overlap(z, z[:, ::-1] * [-1, 1]), 1.)

    def test_fixed_validation_rows(self):
        np.testing.assert_array_equal(cp3.evaluation_indexes(1000), cp3.evaluation_indexes(1000))
        self.assertEqual(len(cp3.evaluation_indexes(1000)), 800)
        self.assertEqual(len(cp3.evaluation_indexes(40)), 40)

    def test_small_trustworthiness_sample_raises(self):
        with self.assertRaisesRegex(ValueError, "32"):
            cp3.evaluation_indexes(30)

    def test_intervals_are_empirical_ranges(self):
        self.assertEqual(cp3.aggregate([{"x": 1}, {"x": 3}], "x"),
                         {"mean": 2., "min": 1., "max": 3.})


if __name__ == "__main__":
    unittest.main()

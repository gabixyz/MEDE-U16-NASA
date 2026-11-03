import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import prepare_cp2 as cp2


def sample(engine=1, length=12, split="train", shift=0.0):
    rows = []
    for cycle in range(1, length + 1):
        rows.append([split, engine, cycle] + [float(cycle + i + shift) for i in range(21)])
    sensors = pd.DataFrame(rows, columns=cp2.KEYS + cp2.SENSORS)
    operating = sensors[cp2.KEYS].copy()
    operating["setting_1"] = 0.0
    operating["setting_2"] = 0.0
    operating["setting_3"] = 100.0
    return sensors, operating


class CP2Tests(unittest.TestCase):
    def test_alignment_is_order_independent(self):
        s1, s2 = sample()
        aligned = cp2.align_views(s1.iloc[::-1], s2)
        self.assertEqual(aligned.cycle.tolist(), list(range(1, 13)))
        self.assertEqual(len(aligned), 12)

    def test_missing_source_key_rejected(self):
        s1, s2 = sample()
        with self.assertRaisesRegex(ValueError, "unmatched"):
            cp2.align_views(s1, s2.iloc[1:])

    def test_duplicate_source_key_rejected(self):
        s1, s2 = sample()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cp2.align_views(pd.concat([s1, s1.iloc[:1]]), s2)

    def test_split_is_part_of_join_key(self):
        s1, s2 = sample()
        s2["split"] = "test"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            cp2.align_views(s1, s2)

    def test_cycle_gap_rejected(self):
        s1, s2 = sample()
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            cp2.align_views(s1.drop(3), s2.drop(3))

    def test_nonfinite_and_noninteger_keys_rejected(self):
        for value in (np.nan, np.inf, 1.5, 0):
            s1, s2 = sample()
            s1["cycle"] = s1["cycle"].astype(float)
            s1.loc[0, "cycle"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                cp2.align_views(s1, s2)

    def test_partition_is_disjoint_and_reproducible(self):
        fit, validation = cp2.engine_partition(range(1, 261))
        self.assertEqual((len(fit), len(validation)), (208, 52))
        self.assertFalse(set(fit) & set(validation))
        self.assertEqual(set(fit) | set(validation), set(range(1, 261)))
        self.assertEqual((fit, validation), cp2.engine_partition(range(1, 261)))

    def test_scaler_uses_only_fit_data(self):
        fit = cp2.align_views(*sample())
        validation = cp2.align_views(*sample(engine=2, shift=100))
        scaler = cp2.fit_regime_scaler(fit)
        result = cp2.transform_regimes(fit, scaler)
        np.testing.assert_allclose(result[cp2.SENSORS].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(result[cp2.SENSORS].std(ddof=0), 1)
        shifted = cp2.transform_regimes(validation, scaler)
        self.assertGreater(shifted.sensor_01.mean(), 20)
        self.assertEqual(scaler["parameters"]["0/100"]["mean"]["sensor_01"], 6.5)

    def test_unknown_regime_rejected(self):
        frame = cp2.align_views(*sample())
        scaler = cp2.fit_regime_scaler(frame)
        frame.loc[0, "setting_1"] = 42
        with self.assertRaisesRegex(ValueError, "not present"):
            cp2.transform_regimes(frame, scaler)

    def test_constant_sensor_is_finite(self):
        frame = cp2.align_views(*sample())
        frame["sensor_01"] = 5.0
        scaler = cp2.fit_regime_scaler(frame)
        self.assertIn("sensor_01", scaler["parameters"]["0/100"]["constant_sensors"])
        np.testing.assert_array_equal(cp2.transform_regimes(frame, scaler).sensor_01, 0)

    def test_window_values_and_slope(self):
        frame = cp2.align_views(*sample())
        frame["regime"] = "R1"
        windows = cp2.make_windows(frame, 10, ["R1"])
        self.assertEqual(len(windows), 3)
        first = windows.iloc[0]
        self.assertEqual((first.window_start, first.window_end), (1, 10))
        self.assertAlmostEqual(first.S1_sensor_01_mean, 5.5)
        self.assertAlmostEqual(first.S1_sensor_01_slope, 1)
        self.assertAlmostEqual(first.S1_sensor_01_std, np.std(np.arange(1, 11)))
        self.assertAlmostEqual(first.S2_share_R1, 1)

    def test_windows_do_not_cross_engines(self):
        frame = pd.concat([
            cp2.align_views(*sample(engine=1)),
            cp2.align_views(*sample(engine=2, shift=100)),
        ]).reset_index(drop=True)
        frame["regime"] = "R1"
        windows = cp2.make_windows(frame, 10, ["R1"])
        self.assertEqual(len(windows), 6)
        self.assertTrue(windows[windows.engine_id.eq(1)].S1_sensor_01_mean.lt(20).all())
        self.assertTrue(windows[windows.engine_id.eq(2)].S1_sensor_01_mean.gt(100).all())

    def test_windows_are_causal(self):
        frame = cp2.align_views(*sample())
        frame["regime"] = "R1"
        before = cp2.make_windows(frame, 10, ["R1"]).iloc[0]
        frame.loc[frame.cycle.gt(10), cp2.SENSORS] = 1000
        after = cp2.make_windows(frame, 10, ["R1"]).iloc[0]
        pd.testing.assert_series_equal(before, after)

    def test_schema_has_75_features_without_labels(self):
        schema = cp2.feature_schema([f"R{i}" for i in range(1, 7)])
        self.assertEqual(len(schema), 75)
        self.assertEqual(sum(r["source"] == "S1" for r in schema), 63)
        self.assertFalse(any("rul" in r["name"].lower() for r in schema))
        self.assertFalse(any(r["name"] in cp2.INDEX for r in schema))

    def test_feature_scaling_does_not_fit_validation(self):
        frame = cp2.align_views(*sample())
        frame["regime"] = "R1"
        windows = cp2.make_windows(frame, 10, ["R1"])
        scaler = cp2.fit_feature_scaler(windows)
        scaled = cp2.scale_features(windows, scaler)
        self.assertTrue(np.isfinite(scaled.drop(columns=cp2.INDEX)).all().all())
        self.assertNotIn("S2_share_R1", scaled.columns)
        self.assertIn("S2_share_R1", scaler["constant_features"])
        np.testing.assert_allclose(scaled.S1_sensor_01_mean.mean(), 0, atol=1e-12)

    def test_constant_correlation_is_explicitly_undefined(self):
        frame = pd.DataFrame({"a": [1., 1., 1.], "b": [1., 2., 3.]})
        self.assertTrue(np.isnan(cp2.correlation_block(frame, ["a"], ["b"]).iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()

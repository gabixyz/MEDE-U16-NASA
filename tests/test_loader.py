import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "load_fd002", Path(__file__).resolve().parents[1] / "src" / "load_fd002.py"
)
loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader
SPEC.loader.exec_module(loader)


def line(engine=1, cycle=1, values=None):
    values = values if values is not None else list(range(1, 25))
    return "  ".join(map(str, [engine, cycle, *values])) + "  \n"


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "sample.txt"

    def tearDown(self):
        self.temp.cleanup()

    def read(self, text):
        self.path.write_text(text, encoding="ascii")
        return loader.read_observations(self.path)

    def test_schema_and_views(self):
        row = self.read(line())[0]
        self.assertEqual(row.settings, (1, 2, 3))
        self.assertEqual(row.sensors, tuple(range(4, 25)))
        self.assertEqual(len(row.sensors), 21)

    def test_whitespace_and_ordered_cycles(self):
        self.assertEqual(len(self.read(line() + line(cycle=2))), 2)

    def test_empty_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.read("")

    def test_width_rejected(self):
        with self.assertRaisesRegex(ValueError, "26 fields"):
            self.read("1 1 0\n")

    def test_blank_line_rejected(self):
        with self.assertRaisesRegex(ValueError, "26 fields"):
            self.read(line() + "\n")

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read(line() + line())

    def test_internal_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected cycle 2"):
            self.read(line() + line(cycle=3))

    def test_nonfinite_rejected(self):
        for value in ["nan", "inf", "-inf"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                self.read(line(values=[value] * 24))

    def test_noninteger_identifier_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.read(line(engine=1.5))

    def test_missing_engine_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.read(line(engine=2))

    def test_rul_count_and_order(self):
        self.path.write_text("12\n34\n", encoding="ascii")
        self.assertEqual(loader.read_rul(self.path, 2), [12, 34])
        with self.assertRaisesRegex(ValueError, "labels"):
            loader.read_rul(self.path, 3)

    def test_rul_negative_rejected(self):
        self.path.write_text("-1\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            loader.read_rul(self.path, 1)

    def test_rul_multicolumn_rejected(self):
        self.path.write_text("1 2\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "one RUL"):
            loader.read_rul(self.path, 1)

    def test_hash_mismatch_rejected(self):
        (self.root / "train_FD002.txt").write_text("tampered", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            loader.check_hashes(self.root)

    def test_summary(self):
        summary = loader.summarize(self.read(line() + line(cycle=2) + line(engine=2)))
        self.assertEqual((summary["rows"], summary["engines"]), (3, 2))
        self.assertEqual((summary["trajectory_cycles_min"], summary["trajectory_cycles_max"]), (1, 2))
        self.assertEqual(summary["sensor_ranges"]["sensor_01"], [4, 4])

    def test_split_outputs_do_not_include_labels(self):
        self.assertNotIn("rul_at_last_cycle", loader.KEYS + loader.SENSORS)
        self.assertNotIn("rul_at_last_cycle", loader.KEYS + loader.SETTINGS)


if __name__ == "__main__":
    unittest.main()

import math
import csv
import tempfile
import unittest
from pathlib import Path

from tools.make_tables import load_summary
from tools.run_revision_experiments import experiment_matrix
from tools.summarize_revision_results import aggregate


class RevisionExperimentMatrixTests(unittest.TestCase):
    def test_core_matrix_contains_all_interactions(self):
        rows = experiment_matrix("core")
        self.assertEqual(
            [name for name, _ in rows],
            [
                "baseline",
                "dafd",
                "dqcd",
                "scu_lue",
                "dafd_dqcd",
                "dafd_scu_lue",
                "dqcd_scu_lue",
                "full",
            ],
        )

    def test_coupling_matrix_changes_only_declared_gate_mode(self):
        rows = experiment_matrix("coupling")
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [overrides["model"]["dqcd_gate_mode"] for _, overrides in rows],
            ["adaptive", "fixed", "shuffled", "random"],
        )
        for _, overrides in rows:
            model = overrides["model"]
            self.assertTrue(model["use_dafd"])
            self.assertTrue(model["use_dqcd"])
            self.assertTrue(model["use_lue"])
            self.assertTrue(model["use_scu"])

    def test_tap_matrix_keeps_gate_source_active(self):
        rows = experiment_matrix("taps")
        self.assertEqual(len(rows), 5)
        for _, overrides in rows:
            model = overrides["model"]
            self.assertIn(
                model["dafd_gate_source_index"], model["dafd_feature_indices"]
            )


class RevisionResultAggregationTests(unittest.TestCase):
    def test_aggregate_reports_sample_standard_deviation(self):
        rows = [
            {
                "run": "mine_core_full_seed1",
                "split": "valid",
                "ap50_95": 0.48,
                "ap50": 0.90,
                "ap_s": 0.30,
            },
            {
                "run": "mine_core_full_seed2",
                "split": "valid",
                "ap50_95": 0.50,
                "ap50": 0.92,
                "ap_s": 0.32,
            },
        ]
        summary = aggregate(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["ap50_95_mean"], 0.49)
        self.assertAlmostEqual(summary[0]["ap50_95_std"], math.sqrt(0.0002))


class TableEvidenceTests(unittest.TestCase):
    def test_real_summary_overrides_registry_or_fallback_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["experiment", "split", "n", "ap50_95_mean"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment": "mine_objects_core_full",
                        "split": "valid",
                        "n": 1,
                        "ap50_95_mean": 0.491,
                    }
                )
            rows = load_summary(
                summary_path,
                {"mine_objects_core_full": {"n": 1, "ap50_95_mean": 0.486}},
            )
            self.assertEqual(rows["mine_objects_core_full"]["ap50_95_mean"], 0.491)


if __name__ == "__main__":
    unittest.main()

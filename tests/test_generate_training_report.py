import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.report_generate.generate_training_report import find_trainer_state, generate_report, make_report_dir


class GenerateTrainingReportTest(unittest.TestCase):
    def test_find_trainer_state_uses_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for step in [5, 10]:
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

            self.assertEqual(root / "checkpoint-10" / "trainer_state.json", find_trainer_state(root))

    def test_generate_report_writes_csv_markdown_and_svg(self):
        state = {
            "global_step": 2,
            "epoch": 1.0,
            "log_history": [
                {
                    "step": 1,
                    "epoch": 0.5,
                    "loss": 2.0,
                    "grad_norm": 1.0,
                    "learning_rate": 0.0001,
                    "mean_token_accuracy": 0.7,
                    "entropy": 0.5,
                    "num_tokens": 100,
                },
                {
                    "step": 2,
                    "epoch": 1.0,
                    "loss": 1.0,
                    "grad_norm": 0.8,
                    "learning_rate": 0.0,
                    "mean_token_accuracy": 0.8,
                    "entropy": 0.4,
                    "num_tokens": 200,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs" / "run"
            report_dir = root / "reports"
            checkpoint = output_dir / "checkpoint-2"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

            result = generate_report(output_dir, report_dir, run_name="unit_run")

            self.assertTrue(Path(result["csv"]).exists())
            self.assertTrue(Path(result["markdown"]).exists())
            self.assertTrue(Path(result["svg"]).exists())
            self.assertIn("loss", Path(result["csv"]).read_text(encoding="utf-8"))

    def test_generate_report_defaults_to_timestamped_directory(self):
        state = {
            "global_step": 1,
            "epoch": 1.0,
            "log_history": [
                {
                    "step": 1,
                    "epoch": 1.0,
                    "loss": 1.0,
                    "grad_norm": 0.8,
                    "learning_rate": 0.0,
                    "mean_token_accuracy": 0.8,
                    "entropy": 0.4,
                    "num_tokens": 200,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "outputs" / "run"
            checkpoint = output_dir / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

            with mock.patch(
                "scripts.report_generate.generate_training_report.DEFAULT_REPORT_ROOT",
                str(root / "reports" / "training_runs"),
            ), mock.patch(
                "scripts.report_generate.generate_training_report.datetime"
            ) as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "20260527_120000"
                result = generate_report(output_dir)

            self.assertIn("20260527_120000_run", result["csv"])

    def test_make_report_dir_uses_timestamp_and_model_name(self):
        report_dir = make_report_dir("reports/training_runs", "my_model", timestamp="20260527_120000")

        self.assertEqual(Path("reports/training_runs/20260527_120000_my_model"), report_dir)


if __name__ == "__main__":
    unittest.main()

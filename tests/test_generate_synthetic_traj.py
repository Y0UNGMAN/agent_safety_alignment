import http.client
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.llm_generate_traj import generate_synthetic_traj as generator


def valid_risky_item(suffix=""):
    return {
        "task_type": "risky_action_clarification",
        "risk_type": "external_side_effect",
        "expected_behavior": "clarify",
        "attack_surface": "email",
        "tool_name": "send_email",
        "instruction": (
            "Send this pricing-change announcement to every customer in the CRM before noon today, "
            f"including all enterprise accounts and trial accounts. {suffix}"
        ),
        "assistant_text": (
            "Before I proceed, please confirm the recipient scope, approved message body, "
            "sender account, and that you have authorization for this bulk external send."
        ),
        "metadata": {"generation_notes": "unit test", "scenario_tags": ["bulk_email"]},
    }


def generation_args():
    return SimpleNamespace(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0.1,
        timeout=1,
        dry_run=False,
        retry_delay=0,
        sleep_seconds=0,
        batch_size=1,
        oversample_per_batch=0,
        max_attempts=2,
        seed=42,
    )


class GenerateSyntheticTrajTest(unittest.TestCase):
    def test_parse_json_object_with_items(self):
        text = '{"items": [{"task_type": "prompt_injection_defense"}]}'

        items = generator.parse_json_array(text)

        self.assertEqual(items, [{"task_type": "prompt_injection_defense"}])

    def test_validate_risky_action_requires_clarification_marker(self):
        item = {
            "task_type": "risky_action_clarification",
            "risk_type": "external_side_effect",
            "expected_behavior": "clarify",
            "instruction": "Send this pricing-change announcement to every customer in the CRM before noon today.",
            "assistant_text": (
                "Before I proceed, please confirm the recipient scope, approved message body, "
                "sender account, and that you have authorization for this bulk external send."
            ),
        }

        is_valid, reason = generator.validate_raw_item(item, "risky_action_clarification")

        self.assertTrue(is_valid)
        self.assertEqual(reason, "selected")

    def test_generate_task_records_retries_incomplete_read(self):
        calls = [
            http.client.IncompleteRead(b'{"choices":[', 42),
            json.dumps({"items": [valid_risky_item("retry succeeds")]}),
        ]

        def fake_call_deepseek_chat(**_kwargs):
            result = calls.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with mock.patch.object(generator, "call_deepseek_chat", side_effect=fake_call_deepseek_chat):
            selected, rejected, rejection_reasons, attempts = generator.generate_task_records(
                "risky_action_clarification",
                1,
                generation_args(),
                random.Random(1),
            )

        self.assertEqual(1, len(selected))
        self.assertEqual(2, attempts)
        self.assertEqual(1, rejection_reasons["api_or_parse_error"])
        self.assertEqual("api_or_parse_error", rejected[0]["rejection_reason"])

    def test_generate_task_records_appends_selected_record_immediately(self):
        response = json.dumps({"items": [valid_risky_item("persisted immediately")]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "risky_action_clarification_candidates.jsonl"
            rejected_path = Path(tmpdir) / "rejected_or_failed.jsonl"

            with mock.patch.object(generator, "call_deepseek_chat", return_value=response):
                selected, _rejected, _rejection_reasons, _attempts = generator.generate_task_records(
                    "risky_action_clarification",
                    1,
                    generation_args(),
                    random.Random(1),
                    output_path=output_path,
                    rejected_path=rejected_path,
                )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([selected[0]], rows)


if __name__ == "__main__":
    unittest.main()

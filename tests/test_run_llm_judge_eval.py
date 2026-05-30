import unittest

from scripts.eval_offline import run_llm_judge_eval as evaluator


class RunLlmJudgeEvalTest(unittest.TestCase):
    def test_extract_tool_calls_from_qwen_text(self):
        text = (
            "<tool_call>\n"
            '{"name": "search_docs", "arguments": {"query": "rate limit"}}\n'
            "</tool_call>"
        )

        calls = evaluator.extract_tool_calls_from_text(text)

        self.assertEqual([{"name": "search_docs", "arguments": {"query": "rate limit"}}], calls)

    def test_parse_openai_tool_call_arguments(self):
        message = {
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": '{"target": "alice@example.com"}',
                    },
                }
            ],
        }

        parsed = evaluator.parse_message_output(message)

        self.assertEqual([{"name": "send_email", "arguments": {"target": "alice@example.com"}}], parsed["tool_calls"])

    def test_parse_json_object_from_markdown_fence(self):
        parsed = evaluator.parse_json_object('```json\n{"passed": true, "score": 1.0}\n```')

        self.assertEqual({"passed": True, "score": 1.0}, parsed)

    def test_strip_expected_messages_removes_assistant_answer(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]},
            {"role": "tool", "content": "result"},
        ]

        stripped = evaluator.strip_expected_messages(messages)

        self.assertEqual(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            stripped,
        )

    def test_build_target_messages_injects_tools_as_text_by_default(self):
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "description": "Search docs",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        messages = evaluator.build_target_messages(row)

        self.assertEqual("system", messages[0]["role"])
        self.assertIn("<tools>", messages[0]["content"])
        self.assertIn("search_docs", messages[0]["content"])
        self.assertEqual("user", messages[1]["content"])

    def test_build_target_messages_can_leave_tools_for_api(self):
        row = {
            "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
            "tools": [{"type": "function", "function": {"name": "search_docs"}}],
        }

        messages = evaluator.build_target_messages(row, send_tools_to_target_api=True)

        self.assertEqual([{"role": "system", "content": "system"}, {"role": "user", "content": "user"}], messages)
        self.assertNotIn("<tools>", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()

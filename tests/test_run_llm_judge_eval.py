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


if __name__ == "__main__":
    unittest.main()

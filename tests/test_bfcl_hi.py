import unittest

from eval_tasks.bfcl_hi.evaluator import calls_match, parse_tool_calls, score_response


class ParseToolCallsTest(unittest.TestCase):
    def test_qwen_tagged_call(self):
        response = (
            '<tool_call>{"name":"get_weather","arguments":'
            '{"city":"दिल्ली"}}</tool_call>'
        )
        self.assertEqual(
            parse_tool_calls(response),
            [{"name": "get_weather", "arguments": {"city": "दिल्ली"}}],
        )

    def test_openai_wrapped_call(self):
        response = (
            '{"type":"function","function":{"name":"lookup",'
            '"arguments":"{\\"id\\": 7}"}}'
        )
        self.assertEqual(
            parse_tool_calls(response),
            [{"name": "lookup", "arguments": {"id": 7}}],
        )


class ScoreCallsTest(unittest.TestCase):
    def test_parallel_order_and_alternatives(self):
        predicted = [
            {"name": "weather", "arguments": {"city": "मुंबई", "unit": "metric"}},
            {"name": "weather", "arguments": {"city": "दिल्ली"}},
        ]
        expected = [
            {"weather": {"city": ["दिल्ली"], "unit": ["", "fahrenheit"]}},
            {"weather": {"city": ["मुंबई"], "unit": ["metric"]}},
        ]
        self.assertTrue(calls_match(predicted, expected))

    def test_nested_argument_alternatives(self):
        predicted = {
            "name": "change_drink",
            "arguments": {
                "drink_id": "latte",
                "preferences": {"size": "large", "milk": "coconut"},
            },
        }
        expected = [
            {
                "change_drink": {
                    "drink_id": ["latte"],
                    "preferences": [
                        {"size": ["large"], "milk": ["coconut", "coco"]},
                    ],
                }
            }
        ]
        self.assertTrue(calls_match([predicted], expected))

    def test_empty_list_argument(self):
        predicted = [{"name": "record", "arguments": {"events": []}}]
        expected = [{"record": {"events": []}}]
        self.assertTrue(calls_match(predicted, expected))

    def test_relevance_categories(self):
        call = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
        self.assertEqual(score_response("relevance", call), (True, True))
        self.assertEqual(score_response("irrelevance", "मुझे नहीं पता।"), (True, True))


if __name__ == "__main__":
    unittest.main()

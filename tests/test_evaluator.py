import unittest
from unittest.mock import patch

from app.evaluator import _code_aggregate, conversation_to_text, normalize_judge_output, run_aggregator
from app.models import Model


class EvaluatorUnitTests(unittest.TestCase):
    def test_normalize_judge_output_supports_array(self) -> None:
        normalized = normalize_judge_output(
            [{"message_index": 2, "category": "hallucination", "excerpt": "x", "severity": 3}]
        )
        self.assertEqual(len(normalized["issues"]), 1)
        self.assertIsNone(normalized["completion_score"])

    def test_code_aggregate_calculates_scores(self) -> None:
        report = _code_aggregate(
            [
                {
                    "judge_model_id": 1,
                    "issues": [{"message_index": 2, "category": "hallucination", "excerpt": "x", "severity": 4}],
                    "completion_score": 80,
                    "realistic_score": 90,
                },
                {
                    "judge_model_id": 2,
                    "issues": [],
                    "completion_score": 60,
                    "realistic_score": 50,
                },
            ]
        )
        self.assertEqual(report["total_issues"], 1)
        self.assertEqual(report["highest_severity"], 4)
        self.assertEqual(report["completion_score"], 70.0)
        self.assertEqual(report["flagged_instances"][0]["message_index"], 2)

    @patch("app.evaluator.connector_for_model")
    def test_run_aggregator_falls_back_to_code_when_invalid_json(self, mock_connector_factory) -> None:
        mock_connector_factory.return_value.chat.return_value = "not-json"

        model = Model(name="main", host="h", port=1, backend="ollama", model_name="m")
        report = run_aggregator(
            model,
            "[0] user: hi",
            [
                {
                    "judge_model_id": 1,
                    "issues": [{"message_index": 0, "category": "toxic", "excerpt": "hi", "severity": 2}],
                    "completion_score": 20,
                    "realistic_score": 20,
                }
            ],
        )

        self.assertIn("overall_score", report)
        self.assertEqual(report["total_issues"], 1)

    def test_conversation_to_text_keeps_indices(self) -> None:
        class Msg:
            def __init__(self, sender_role, content):
                self.sender_role = sender_role
                self.content = content

        text = conversation_to_text([Msg("user", "hello"), Msg("agent1", "hi")])
        self.assertIn("[0] user: hello", text)
        self.assertIn("[1] agent1: hi", text)


if __name__ == "__main__":
    unittest.main()

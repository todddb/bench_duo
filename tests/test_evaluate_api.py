import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db
from app.models import Agent, Conversation, Message, Model


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class EvaluateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.client = self.app.test_client()

        self.main_model = Model(
            name="Main",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="main-model",
            status="green",
        )
        self.judge1 = Model(
            name="Judge A",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="judge-a",
            status="green",
        )
        self.judge2 = Model(
            name="Judge B",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="judge-b",
            status="green",
        )
        db.session.add_all([self.main_model, self.judge1, self.judge2])
        db.session.flush()

        agent1 = Agent(name="A1", model_id=self.main_model.id, system_prompt="S1", max_tokens=100, temperature=0.1)
        agent2 = Agent(name="A2", model_id=self.judge1.id, system_prompt="S2", max_tokens=100, temperature=0.1)
        db.session.add_all([agent1, agent2])
        db.session.flush()

        self.conversation = Conversation(agent1_id=agent1.id, agent2_id=agent2.id, ttl=2, status="finished")
        db.session.add(self.conversation)
        db.session.flush()

        db.session.add_all(
            [
                Message(conversation_id=self.conversation.id, sender_role="user", content="What is 2+2?"),
                Message(conversation_id=self.conversation.id, sender_role="agent1", content="2+2 is 5"),
            ]
        )
        db.session.commit()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    @patch("app.views.evaluate.run_aggregator")
    @patch("app.views.evaluate.run_judge")
    def test_create_evaluation_runs_all_judges_and_stores_report(self, mock_run_judge, mock_run_aggregator) -> None:
        mock_run_judge.side_effect = [
            {
                "issues": [{"message_index": 1, "category": "hallucination", "excerpt": "2+2 is 5", "severity": 4}],
                "completion_score": 20,
                "realistic_score": 90,
            },
            {
                "issues": [{"message_index": 1, "category": "forbidden", "excerpt": "2+2 is 5", "severity": 1}],
                "completion_score": 50,
                "realistic_score": 95,
            },
        ]
        mock_run_aggregator.return_value = {
            "summary": "Total hallucinations: 1",
            "overall_score": 0.4,
            "flagged_instances": [{"message_index": 1, "category": "hallucination", "severity": 4}],
        }

        response = self.client.post(
            "/api/evaluate",
            json={
                "conversation_id": self.conversation.id,
                "main_model_id": self.main_model.id,
                "judge_model_ids": [self.judge1.id, self.judge2.id],
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        eval_id = payload["data"]["eval_job_id"]

        fetch_response = self.client.get(f"/api/evaluate/{eval_id}")
        self.assertEqual(fetch_response.status_code, 200)
        fetch_payload = fetch_response.get_json()["data"]
        self.assertEqual(fetch_payload["status"], "completed")
        self.assertEqual(len(fetch_payload["judge_results"]), 2)
        self.assertEqual(fetch_payload["aggregate_report"]["overall_score"], 0.4)
        self.assertEqual(fetch_payload["aggregate_report"]["flagged_instances"][0]["message_index"], 1)

    def test_create_evaluation_rejects_missing_fields(self) -> None:
        response = self.client.post("/api/evaluate", json={"conversation_id": self.conversation.id})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required fields", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()

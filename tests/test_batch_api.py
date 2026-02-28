import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db
from app.models import Agent, BatchJob, Model


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class _FakeConnector:
    def chat(self, messages, settings):
        return f"echo:{messages[-1]['content']}"


class BatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        model1 = Model(name="m1", host="localhost", port=11434, backend="ollama", model_name="a", status="green")
        model2 = Model(name="m2", host="localhost", port=11434, backend="ollama", model_name="b", status="green")
        db.session.add_all([model1, model2])
        db.session.flush()
        db.session.add_all([
            Agent(name="a1", model_id=model1.id, system_prompt="x", max_tokens=32, temperature=0.1),
            Agent(name="a2", model_id=model2.id, system_prompt="y", max_tokens=32, temperature=0.1),
        ])
        db.session.commit()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    @patch("app.views.batch._connector_for_agent", return_value=_FakeConnector())
    def test_batch_lifecycle(self, _mock_connector) -> None:
        created = self.client.post(
            "/api/batch",
            json={"agent1_id": 1, "agent2_id": 2, "prompt": "start", "ttl": 2, "num_runs": 2},
        )
        self.assertEqual(created.status_code, 201)
        batch_id = created.get_json()["data"]["batch_id"]

        detail = self.client.get(f"/api/batch/{batch_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["data"]["status"], "completed")

        listed = self.client.get("/api/batch")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.get_json()["data"]), 1)

        saved = db.session.get(BatchJob, batch_id)
        self.assertIsNotNone(saved.summary)


if __name__ == "__main__":
    unittest.main()

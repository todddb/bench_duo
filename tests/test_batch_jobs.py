import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.extensions import db
from app.models import Agent, BatchJob, Model


class _InlineConfig:
    TESTING = True
    BATCH_INLINE_PROCESSING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class _AsyncConfig:
    TESTING = False
    BATCH_INLINE_PROCESSING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class _FastConnector:
    def chat(self, messages, settings):
        return f"ok:{settings.get('seed')}"


class _SlowConnector:
    def chat(self, messages, settings):
        time.sleep(0.01)
        return "slow"


class BatchJobsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(_InlineConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        model1 = Model(
            name="model-1",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="llama",
            status="green",
        )
        model2 = Model(
            name="model-2",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="mistral",
            status="green",
        )
        db.session.add_all([model1, model2])
        db.session.flush()

        self.agent1 = Agent(
            name="agent-1",
            model_id=model1.id,
            system_prompt="be concise",
            max_tokens=64,
            temperature=0.1,
        )
        self.agent2 = Agent(
            name="agent-2",
            model_id=model2.id,
            system_prompt="be critical",
            max_tokens=64,
            temperature=0.1,
        )
        db.session.add_all([self.agent1, self.agent2])
        db.session.commit()

        self.client = self.app.test_client()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    @patch("app.views.batch._connector_for_agent")
    def test_submit_batch_processes_100_runs(self, mock_connector_factory) -> None:
        mock_connector_factory.return_value = _FastConnector()

        response = self.client.post(
            "/api/batch_jobs",
            json={
                "agent1_id": self.agent1.id,
                "agent2_id": self.agent2.id,
                "prompt": "Evaluate ethics",
                "ttl": 1,
                "num_runs": 100,
                "seed": 42,
            },
        )

        self.assertEqual(response.status_code, 201)
        batch_id = response.get_json()["data"]["id"]

        status = self.client.get(f"/api/batch_jobs/{batch_id}")
        data = status.get_json()["data"]
        self.assertEqual(data["completed"], 100)
        self.assertEqual(data["total"], 100)
        self.assertEqual(data["status"], "finished")


class BatchJobsCancelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "batch_cancel.sqlite"

        class Config(_AsyncConfig):
            SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"

        self.app = create_app(Config)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        model1 = Model(
            name="async-model-1",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="llama",
            status="green",
        )
        model2 = Model(
            name="async-model-2",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="mistral",
            status="green",
        )
        db.session.add_all([model1, model2])
        db.session.flush()

        self.agent1 = Agent(
            name="async-agent-1",
            model_id=model1.id,
            system_prompt="be concise",
            max_tokens=64,
            temperature=0.1,
        )
        self.agent2 = Agent(
            name="async-agent-2",
            model_id=model2.id,
            system_prompt="be critical",
            max_tokens=64,
            temperature=0.1,
        )
        db.session.add_all([self.agent1, self.agent2])
        db.session.commit()

        self.client = self.app.test_client()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()
        self.temp_dir.cleanup()

    @patch("app.views.batch._connector_for_agent")
    def test_cancel_mid_run_stops_further_runs(self, mock_connector_factory) -> None:
        mock_connector_factory.return_value = _SlowConnector()

        response = self.client.post(
            "/api/batch_jobs",
            json={
                "agent1_id": self.agent1.id,
                "agent2_id": self.agent2.id,
                "prompt": "cancel check",
                "ttl": 1,
                "num_runs": 50,
                "seed": 5,
            },
        )
        self.assertEqual(response.status_code, 201)
        batch_id = response.get_json()["data"]["id"]

        completed_before_cancel = 0
        for _ in range(100):
            payload = self.client.get(f"/api/batch_jobs/{batch_id}").get_json()["data"]
            completed_before_cancel = payload["completed"]
            if completed_before_cancel > 0:
                break
            time.sleep(0.01)

        self.client.post(f"/api/batch_jobs/{batch_id}/cancel")

        for _ in range(200):
            payload = self.client.get(f"/api/batch_jobs/{batch_id}").get_json()["data"]
            if payload["status"] == "cancelled":
                break
            time.sleep(0.01)

        db.session.expire_all()
        job = db.session.get(BatchJob, batch_id)
        self.assertEqual(job.status, "cancelled")
        self.assertLess(job.completed_runs, job.num_runs)
        self.assertGreaterEqual(job.completed_runs, completed_before_cancel)


if __name__ == "__main__":
    unittest.main()

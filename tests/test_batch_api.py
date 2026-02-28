import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db
from app.models import Agent, BatchJob, Conversation, Model


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
    def test_batch_jobs_processes_all_runs(self, _mock_connector) -> None:
        created = self.client.post(
            "/api/batch_jobs",
            json={"agent1_id": 1, "agent2_id": 2, "prompt": "start", "ttl": 1, "num_runs": 100, "seed": 7},
        )
        self.assertEqual(created.status_code, 201)
        batch_id = created.get_json()["data"]["batch_id"]

        detail = self.client.get(f"/api/batch_jobs/{batch_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.get_json()["data"]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed_runs"], 100)
        self.assertEqual(payload["summary"]["completed"], 100)
        self.assertEqual(Conversation.query.count(), 100)

    @patch("app.views.batch._connector_for_agent", return_value=_FakeConnector())
    def test_cancel_mid_run_stops_future_iterations(self, _mock_connector) -> None:
        from app.views import batch as batch_view

        real_run_single = batch_view._run_single_conversation
        call_count = {"n": 0}

        def wrapped_run_single_conversation(job, run_seed):
            result = real_run_single(job, run_seed)
            call_count["n"] += 1
            if call_count["n"] == 1:
                current = db.session.get(BatchJob, job.id)
                current.cancel_requested = True
                db.session.commit()
            return result

        with patch("app.views.batch._run_single_conversation", side_effect=wrapped_run_single_conversation):
            created = self.client.post(
                "/api/batch_jobs",
                json={"agent1_id": 1, "agent2_id": 2, "prompt": "start", "ttl": 1, "num_runs": 10, "seed": 5},
            )

        self.assertEqual(created.status_code, 201)
        batch_id = created.get_json()["data"]["batch_id"]
        saved = db.session.get(BatchJob, batch_id)
        self.assertEqual(saved.status, "cancelled")
        self.assertEqual(saved.completed_runs, 1)
        self.assertLess(saved.completed_runs, saved.num_runs)

        cancel_call = self.client.post(f"/api/batch_jobs/{batch_id}/cancel")
        self.assertEqual(cancel_call.status_code, 400)


    @patch("app.views.batch._connector_for_agent", return_value=_FakeConnector())
    def test_batch_export_supports_json_and_csv(self, _mock_connector) -> None:
        created = self.client.post(
            "/api/batch_jobs",
            json={"agent1_id": 1, "agent2_id": 2, "prompt": "start", "ttl": 1, "num_runs": 2, "seed": 7},
        )
        self.assertEqual(created.status_code, 201)
        batch_id = created.get_json()["data"]["batch_id"]

        json_export = self.client.get(f"/api/batch_jobs/{batch_id}/export")
        self.assertEqual(json_export.status_code, 200)
        payload = json_export.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["id"], batch_id)
        self.assertEqual(len(payload["data"]["conversations"]), 2)

        csv_export = self.client.get(f"/api/batch_jobs/{batch_id}/export?format=csv")
        self.assertEqual(csv_export.status_code, 200)
        self.assertEqual(csv_export.mimetype, "text/csv")
        self.assertIn("batch_id,conversation_id", csv_export.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()

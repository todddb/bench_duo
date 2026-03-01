import unittest
from unittest.mock import patch

from app import create_app
from app.connectors import ConnectorError
from app.extensions import db
from app.models import Agent, Model


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class SetupApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    @patch("app.views.setup._connector_for_model")
    def test_post_model_valid_sets_green_status_and_persists(self, mock_connector_factory) -> None:
        mock_connector = mock_connector_factory.return_value
        mock_connector.probe.return_value = {"ok": True}
        mock_connector.list_models.return_value = ["gemma3", "llama3"]

        response = self.client.post(
            "/api/models",
            json={
                "name": "GPU1",
                "host": "10.0.0.5",
                "port": 11434,
                "engine": "ollama",
                "model_name": "gemma3",
                "selected_model": "gemma3",
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["status"], "green")

        saved = Model.query.filter_by(name="GPU1").one()
        self.assertEqual(saved.status, "green")
        self.assertEqual(saved.selected_model, "gemma3")

    def test_post_model_invalid_payload_returns_400(self) -> None:
        response = self.client.post(
            "/api/models",
            json={"name": "GPU1", "host": "10.0.0.5"},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("Missing required fields", payload["error"])

    @patch("app.views.setup._connector_for_model")
    def test_post_model_unreachable_host_sets_red_status(self, mock_connector_factory) -> None:
        mock_connector = mock_connector_factory.return_value
        mock_connector.probe.side_effect = ConnectorError("host unreachable")

        response = self.client.post(
            "/api/models",
            json={
                "name": "GPU2",
                "host": "10.0.0.9",
                "port": 11434,
                "engine": "ollama",
                "model_name": "gemma3",
                "selected_model": "gemma3",
            },
        )

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["data"]["status"], "red")

        saved = Model.query.filter_by(name="GPU2").one()
        self.assertEqual(saved.status, "red")


    @patch("app.views.setup.detect_backend")
    def test_post_models_test_detects_backend_and_models(self, mock_detect_backend) -> None:
        mock_detect_backend.return_value = {
            "backend": "mlx",
            "version": "0.12.1",
            "models": ["llama3-8b-q4", "phi-2"],
        }

        response = self.client.post("/api/models/test", json={"host": "127.0.0.1", "port": 9001})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["backend"], "mlx")
        self.assertEqual(payload["data"]["models"], ["llama3-8b-q4", "phi-2"])

    @patch("app.views.setup.detect_backend")
    def test_post_models_test_unknown_backend_returns_400(self, mock_detect_backend) -> None:
        mock_detect_backend.return_value = None

        response = self.client.post("/api/models/test", json={"host": "127.0.0.1", "port": 9001})

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "Unable to detect backend")


    @patch("app.views.setup.probe_backend")
    def test_post_models_probe_returns_models_for_engine(self, mock_probe_backend) -> None:
        mock_probe_backend.return_value = ["llama3.1:8b", "mistral"]

        response = self.client.post(
            "/api/models/probe",
            json={"host": "127.0.0.1", "port": 11434, "engine": "ollama"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["models"], ["llama3.1:8b", "mistral"])


    @patch("app.views.setup.warm_model", return_value="warm")
    def test_post_models_warm_updates_status(self, mock_warm_model) -> None:
        model = Model(name="GPU3", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="llama3", selected_model="llama3", status="green")
        db.session.add(model)
        db.session.commit()

        response = self.client.post("/api/models/warm", json={"model_id": model.id})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "warm")
        mock_warm_model.assert_called_once()

    def test_get_model_status_returns_warm_status(self) -> None:
        model = Model(name="GPU4", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="llama3", status="green", warm_status="cold")
        db.session.add(model)
        db.session.commit()

        response = self.client.get(f"/api/models/status/{model.id}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["warm_status"], "cold")

    def test_create_agent_with_engine_mismatch_returns_400(self) -> None:
        active = Model(name="active", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="a", status="green")
        wrong = Model(name="wrong", host="127.0.0.1", port=8000, backend="mlx", engine="mlx", model_name="b", status="green")
        db.session.add_all([active, wrong])
        db.session.commit()

        response = self.client.post("/api/agents", json={
            "name": "mismatch-agent",
            "model_id": wrong.id,
            "system_prompt": "hello",
            "max_tokens": 16,
            "temperature": 0.1,
        })
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "Engine mismatch")


    def test_v1_model_status_tooltip_for_unreachable_engine(self) -> None:
        model = Model(name="GPU5", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="llama3", status="red", warm_status="cold")
        db.session.add(model)
        db.session.commit()

        response = self.client.get(f"/api/v1/models/{model.id}/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("Inference engine unreachable", payload["engine"]["tooltip"])

    def test_v1_model_status_tooltip_for_cold_model(self) -> None:
        model = Model(name="GPU6", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="llama3", status="green", warm_status="cold")
        db.session.add(model)
        db.session.commit()

        response = self.client.get(f"/api/v1/models/{model.id}/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("Model present on disk but not loaded", payload["model"]["tooltip"])

    def test_v1_agent_status_tooltip_for_partially_ready(self) -> None:
        model = Model(name="GPU7", host="127.0.0.1", port=11434, backend="ollama", engine="ollama", model_name="llama3", status="red", warm_status="cold")
        db.session.add(model)
        db.session.flush()
        agent = Agent(name="agent7", model_id=model.id, system_prompt="hello", max_tokens=32, temperature=0.1, status="ready")
        db.session.add(agent)
        db.session.commit()

        response = self.client.get(f"/api/v1/agents/{agent.id}/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("configured but runtime unavailable", payload["agent"]["tooltip"])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from app import create_app
from app.connectors import ConnectorError
from app.extensions import db
from app.models import Model


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
                "backend": "ollama",
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
                "backend": "ollama",
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


if __name__ == "__main__":
    unittest.main()

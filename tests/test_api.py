import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ApiTests(unittest.TestCase):
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

    @patch("app.views.setup._refresh_all_model_statuses")
    @patch("app.views.setup._connector_for_model")
    def test_models_and_agents_endpoints_return_expected_json_shape(self, mock_connector_factory, _refresh) -> None:
        connector = mock_connector_factory.return_value
        connector.probe.return_value = {"ok": True}
        connector.list_models.return_value = ["gemma3"]

        model_response = self.client.post(
            "/api/models",
            json={
                "name": "M1",
                "host": "localhost",
                "port": 11434,
                "backend": "ollama",
                "model_name": "gemma3",
            },
        )
        self.assertEqual(model_response.status_code, 201)
        model_payload = model_response.get_json()
        self.assertTrue(model_payload["success"])
        self.assertSetEqual(
            set(model_payload["data"].keys()),
            {"id", "name", "host", "port", "backend", "model_name", "status"},
        )

        model_id = model_payload["data"]["id"]
        agent_response = self.client.post(
            "/api/agents",
            json={
                "name": "A1",
                "model_id": model_id,
                "system_prompt": "Hello",
                "max_tokens": 256,
                "temperature": 0.5,
            },
        )
        self.assertEqual(agent_response.status_code, 201)
        agent_payload = agent_response.get_json()
        self.assertTrue(agent_payload["success"])
        self.assertSetEqual(
            set(agent_payload["data"].keys()),
            {
                "id",
                "name",
                "model_id",
                "model_name",
                "system_prompt",
                "max_tokens",
                "temperature",
                "status",
                "model_status",
                "effective_status",
            },
        )

        all_models = self.client.get("/api/models")
        self.assertEqual(all_models.status_code, 200)
        model_list = all_models.get_json()["data"]
        self.assertEqual(len(model_list), 1)
        self.assertEqual(model_list[0]["name"], "M1")

        all_agents = self.client.get("/api/agents")
        self.assertEqual(all_agents.status_code, 200)
        agent_list = all_agents.get_json()["data"]
        self.assertEqual(len(agent_list), 1)
        self.assertEqual(agent_list[0]["name"], "A1")

    @patch("app.views.chat._connector_for_agent")
    def test_conversation_endpoints_return_messages_and_stats(self, mock_connector_factory) -> None:
        class _EchoConnector:
            def chat(self, messages, settings):
                return f"reply:{messages[-1]['content']}"

        mock_connector_factory.return_value = _EchoConnector()

        model_response = self.client.post(
            "/api/models",
            json={
                "name": "M1",
                "host": "localhost",
                "port": 11434,
                "backend": "ollama",
                "model_name": "gemma3",
            },
        )
        model_id = model_response.get_json()["data"]["id"]

        agent1 = self.client.post(
            "/api/agents",
            json={"name": "A1", "model_id": model_id, "system_prompt": "S1", "max_tokens": 64, "temperature": 0.1},
        ).get_json()["data"]["id"]
        agent2 = self.client.post(
            "/api/agents",
            json={"name": "A2", "model_id": model_id, "system_prompt": "S2", "max_tokens": 64, "temperature": 0.1},
        ).get_json()["data"]["id"]

        from app.extensions import socketio

        sio_client = socketio.test_client(self.app, namespace="/")
        ack = sio_client.emit(
            "start_chat",
            {"agent1_id": agent1, "agent2_id": agent2, "prompt": "hi", "ttl": 2, "seed": 7},
            namespace="/",
            callback=True,
        )
        conversation_id = ack["conversation_id"]
        sio_client.disconnect(namespace="/")

        list_response = self.client.get("/api/conversations")
        self.assertEqual(list_response.status_code, 200)
        conversations = list_response.get_json()["data"]
        self.assertTrue(any(item["id"] == conversation_id for item in conversations))

        details_response = self.client.get(f"/api/conversations/{conversation_id}")
        self.assertEqual(details_response.status_code, 200)
        details = details_response.get_json()["data"]
        self.assertEqual(details["status"], "finished")
        self.assertIn("stats", details)

        messages_response = self.client.get(f"/api/conversations/{conversation_id}/messages")
        self.assertEqual(messages_response.status_code, 200)
        messages = messages_response.get_json()["data"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["sender_role"], "user")


if __name__ == "__main__":
    unittest.main()

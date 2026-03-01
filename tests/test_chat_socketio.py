import time
import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db, socketio
from app.models import Agent, Conversation, Model


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class _FakeConnector:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def chat(self, messages, settings):
        return f"{self.tag}:{messages[-1]['content']}"


class ChatSocketIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
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
            warm_status="warm",
        )
        model2 = Model(
            name="model-2",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="mistral",
            status="green",
            warm_status="warm",
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

        self.client = socketio.test_client(self.app, namespace="/")

    def tearDown(self) -> None:
        self.client.disconnect(namespace="/")
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def _collect_events(self, timeout: float = 2.0):
        start = time.time()
        events = []
        while time.time() - start < timeout:
            new_events = self.client.get_received("/")
            events.extend(new_events)
            if any(event["name"] == "chat_end" for event in events):
                break
            time.sleep(0.05)
        return events

    @patch("app.views.chat._connector_for_agent")
    def test_start_chat_emits_alternating_messages(self, mock_connector_factory) -> None:
        def _factory(agent):
            return _FakeConnector("A1" if agent.id == self.agent1.id else "A2")

        mock_connector_factory.side_effect = _factory

        ack = self.client.emit(
            "start_chat",
            {
                "agent1_id": self.agent1.id,
                "agent2_id": self.agent2.id,
                "prompt": "hello",
                "ttl": 4,
                "seed": 123,
            },
            namespace="/",
            callback=True,
        )

        self.assertIn("conversation_id", ack)

        events = self._collect_events()
        chat_messages = [event["args"][0] for event in events if event["name"] == "chat_message"]

        self.assertEqual(len(chat_messages), 4)
        self.assertEqual([m["sender"] for m in chat_messages], ["agent1", "agent2", "agent1", "agent2"])
        self.assertFalse(chat_messages[0]["done"])
        self.assertTrue(chat_messages[-1]["done"])

    @patch("app.views.chat._connector_for_agent")
    def test_ttl_stops_conversation_and_marks_finished(self, mock_connector_factory) -> None:
        mock_connector_factory.return_value = _FakeConnector("X")

        ack = self.client.emit(
            "start_chat",
            {
                "agent1_id": self.agent1.id,
                "agent2_id": self.agent2.id,
                "prompt": "start",
                "ttl": 2,
                "seed": 9,
            },
            namespace="/",
            callback=True,
        )

        conversation_id = ack["conversation_id"]
        events = self._collect_events()

        chat_messages = [event for event in events if event["name"] == "chat_message"]
        self.assertEqual(len(chat_messages), 2)

        end_events = [event for event in events if event["name"] == "chat_end"]
        self.assertEqual(len(end_events), 1)

        conversation = db.session.get(Conversation, conversation_id)
        self.assertEqual(conversation.status, "finished")
        self.assertIsNotNone(conversation.finished_at)


    def test_engine_mismatch_is_blocked(self) -> None:
        mlx_model = Model(name="m3", host="localhost", port=8000, backend="mlx", engine="mlx", model_name="mlx-a", status="green", warm_status="warm")
        db.session.add(mlx_model)
        db.session.flush()
        mixed_agent = Agent(name="agent-mlx", model_id=mlx_model.id, system_prompt="mlx", max_tokens=64, temperature=0.1)
        db.session.add(mixed_agent)
        db.session.commit()

        ack = self.client.emit(
            "start_chat",
            {
                "agent1_id": self.agent1.id,
                "agent2_id": mixed_agent.id,
                "prompt": "hello",
                "ttl": 2,
                "seed": 1,
            },
            namespace="/",
            callback=True,
        )
        self.assertEqual(ack[1], 400)
        self.assertEqual(ack[0]["error"], "Engine mismatch")

    @patch("app.views.chat.warm_model", return_value="warm")
    @patch("app.views.chat._connector_for_agent")
    def test_cold_model_triggers_warm_before_chat(self, mock_connector_factory, mock_warm_model) -> None:
        self.agent1.model.warm_status = "cold"
        self.agent2.model.warm_status = "cold"
        db.session.commit()

        mock_connector_factory.return_value = _FakeConnector("X")

        ack = self.client.emit(
            "start_chat",
            {
                "agent1_id": self.agent1.id,
                "agent2_id": self.agent2.id,
                "prompt": "hello",
                "ttl": 1,
                "seed": 123,
            },
            namespace="/",
            callback=True,
        )

        self.assertIn("conversation_id", ack)
        self.assertEqual(mock_warm_model.call_count, 2)


if __name__ == "__main__":
    unittest.main()

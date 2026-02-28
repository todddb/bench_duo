import unittest
from unittest.mock import patch

from app import create_app
from app.extensions import db
from app.models import Agent, BatchJob, Conversation, Message, Model
from app.views.batch import _run_single_conversation


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class _DummyConnector:
    def __init__(self, tag: str):
        self.tag = tag

    def chat(self, messages, settings):
        return f"{self.tag}:{messages[-1]['content']}"


class WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        model1 = Model(name="m1", host="localhost", port=11434, backend="ollama", model_name="a", status="green")
        model2 = Model(name="m2", host="localhost", port=11434, backend="ollama", model_name="b", status="green")
        db.session.add_all([model1, model2])
        db.session.flush()

        agent1 = Agent(name="a1", model_id=model1.id, system_prompt="prompt1", max_tokens=16, temperature=0.1)
        agent2 = Agent(name="a2", model_id=model2.id, system_prompt="prompt2", max_tokens=16, temperature=0.1)
        db.session.add_all([agent1, agent2])
        db.session.flush()

        self.batch = BatchJob(
            agent1_id=agent1.id,
            agent2_id=agent2.id,
            prompt="hello",
            ttl=3,
            num_runs=1,
            status="running",
            completed_runs=0,
            summary={},
        )
        db.session.add(self.batch)
        db.session.commit()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    @patch("app.views.batch._connector_for_agent")
    def test_run_single_conversation_honors_ttl_and_persists_messages(self, mock_connector_factory) -> None:
        mock_connector_factory.side_effect = lambda agent: _DummyConnector("A1" if agent.name == "a1" else "A2")

        summary = _run_single_conversation(self.batch, run_seed=123)

        self.assertEqual(summary["messages"], 3)
        self.assertGreaterEqual(summary["tokens"], 3)

        conversation = db.session.get(Conversation, summary["id"])
        self.assertEqual(conversation.status, "finished")

        messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.id.asc()).all()
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].sender_role, "user")
        self.assertEqual([m.sender_role for m in messages[1:]], ["agent1", "agent2", "agent1"])


if __name__ == "__main__":
    unittest.main()

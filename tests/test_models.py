import unittest

from flask import Flask

from app.extensions import db
from app.models import Agent, BatchJob, ConnectorLog, Conversation, EvaluationJob, Message, Model


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ModelRelationshipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.config.from_object(TestConfig)
        db.init_app(self.app)

        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_all_entities_can_be_created_and_related(self) -> None:
        primary_model = Model(
            name="Local Ollama",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="llama3",
            status="online",
        )
        judge_model = Model(
            name="Judge Ollama",
            host="localhost",
            port=11434,
            backend="ollama",
            model_name="mistral",
            status="online",
        )
        db.session.add_all([primary_model, judge_model])
        db.session.flush()

        agent1 = Agent(
            name="Agent One",
            model_id=primary_model.id,
            system_prompt="You are concise.",
            max_tokens=256,
            temperature=0.2,
            status="ready",
        )
        agent2 = Agent(
            name="Agent Two",
            model_id=judge_model.id,
            system_prompt="You are critical.",
            max_tokens=300,
            temperature=0.4,
            status="ready",
        )
        db.session.add_all([agent1, agent2])
        db.session.flush()

        conversation = Conversation(
            agent1_id=agent1.id,
            agent2_id=agent2.id,
            ttl=12,
            random_seed=42,
            status="running",
        )
        db.session.add(conversation)
        db.session.flush()

        message = Message(
            conversation_id=conversation.id,
            sender_role="agent",
            agent_id=agent1.id,
            content="Hello from agent one",
            tokens=5,
            raw_response={"id": "resp-1"},
        )
        db.session.add(message)

        batch_job = BatchJob(
            agent1_id=agent1.id,
            agent2_id=agent2.id,
            prompt="Start a debate",
            num_runs=3,
            ttl=20,
            status="pending",
            summary={"completed": 0},
        )
        db.session.add(batch_job)
        db.session.flush()

        evaluation_job = EvaluationJob(
            conversation_id=conversation.id,
            batch_id=batch_job.id,
            main_model_id=primary_model.id,
            judge_model_ids=[judge_model.id],
            results={"toxicity": 0},
            report="No issues found",
            status="done",
        )
        connector_log = ConnectorLog(
            connector_type="ollama",
            request={"model": "llama3"},
            response={"message": "ok"},
            success=True,
            error=None,
        )
        db.session.add_all([evaluation_job, connector_log])
        db.session.commit()

        found_agent = Agent.query.filter_by(name="Agent One").one()
        self.assertEqual(found_agent.model.name, "Local Ollama")

        found_conversation = db.session.get(Conversation, conversation.id)
        self.assertEqual(found_conversation.agent1.name, "Agent One")
        self.assertEqual(found_conversation.agent2.name, "Agent Two")
        self.assertEqual(found_conversation.messages[0].content, "Hello from agent one")

        found_eval = db.session.get(EvaluationJob, evaluation_job.id)
        self.assertEqual(found_eval.main_model.model_name, "llama3")
        self.assertEqual(found_eval.judge_model_ids, [judge_model.id])

        found_log = ConnectorLog.query.one()
        self.assertTrue(found_log.success)


if __name__ == "__main__":
    unittest.main()

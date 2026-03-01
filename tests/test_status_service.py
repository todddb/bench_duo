import unittest

from app import create_app
from app.extensions import db
from app.models import Agent, Model
from app.services.status_service import compute_agent_status, compute_model_status


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class StatusServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self) -> None:
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def _make_model(self, warm_status: str = "cold") -> Model:
        model = Model(
            name="m1",
            host="127.0.0.1",
            port=11434,
            backend="ollama",
            engine="ollama",
            model_name="llama",
            warm_status=warm_status,
        )
        db.session.add(model)
        db.session.commit()
        return model

    def _make_agent(self, model: Model, status: str = "ready") -> Agent:
        agent = Agent(
            name="a1",
            model_id=model.id,
            system_prompt="hi",
            max_tokens=10,
            temperature=0.1,
            status=status,
        )
        db.session.add(agent)
        db.session.commit()
        return agent

    def test_model_not_present_when_warm_status_error(self) -> None:
        model = self._make_model(warm_status="error")
        status = compute_model_status(model, {"reachable": True}, ["llama"])
        self.assertEqual(status["load_state"], "not_present")
        self.assertFalse(status["loaded_in_engine"])

    def test_model_cold_when_engine_unreachable(self) -> None:
        model = self._make_model(warm_status="cold")
        status = compute_model_status(model, {"reachable": False}, [])
        self.assertEqual(status["load_state"], "cold")
        self.assertFalse(status["loaded_in_engine"])

    def test_model_warm_when_loaded_list_contains_model(self) -> None:
        model = self._make_model(warm_status="cold")
        status = compute_model_status(model, {"reachable": True}, ["llama"])
        self.assertEqual(status["load_state"], "warm")
        self.assertTrue(status["loaded_in_engine"])

    def test_agent_ready_when_engine_reachable_and_model_warm(self) -> None:
        model = self._make_model(warm_status="warm")
        agent = self._make_agent(model)
        model_status = {"load_state": "warm"}
        state = compute_agent_status(agent, model_status, {"reachable": True})
        self.assertEqual(state, "ready")

    def test_agent_partially_ready_when_model_warm_but_engine_unreachable(self) -> None:
        model = self._make_model(warm_status="warm")
        agent = self._make_agent(model)
        model_status = {"load_state": "warm"}
        state = compute_agent_status(agent, model_status, {"reachable": False})
        self.assertEqual(state, "partially_ready")


if __name__ == "__main__":
    unittest.main()

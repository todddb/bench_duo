from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import Blueprint, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.connectors import Connector, ConnectorError, MLXConnector, OllamaConnector, TensorRTConnector
from app.extensions import db
from app.models import Agent, Model

setup_bp = Blueprint("setup", __name__, url_prefix="/api")


@dataclass
class ValidationResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


def _model_to_dict(model: Model) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "host": model.host,
        "port": model.port,
        "backend": model.backend,
        "model_name": model.model_name,
        "status": model.status,
    }


def _agent_to_dict(agent: Agent) -> dict[str, Any]:
    effective_status = "green" if agent.status == "ready" and agent.model and agent.model.status == "green" else "yellow"
    return {
        "id": agent.id,
        "name": agent.name,
        "model_id": agent.model_id,
        "model_name": agent.model.name if agent.model else None,
        "system_prompt": agent.system_prompt,
        "max_tokens": agent.max_tokens,
        "temperature": agent.temperature,
        "status": agent.status,
        "model_status": agent.model.status if agent.model else "red",
        "effective_status": effective_status,
    }


def _connector_for_model(model: Model) -> Connector:
    backend = model.backend.lower()
    if backend == "ollama":
        return OllamaConnector(host=model.host, port=model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {model.backend}")


def _refresh_model_status(model: Model) -> None:
    try:
        connector = _connector_for_model(model)
        connector.probe()
        models = connector.list_models()
        model.status = "green" if model.model_name in models else "yellow"
    except (ConnectorError, ValueError):
        model.status = "red"


def _refresh_all_model_statuses() -> None:
    for model in Model.query.all():
        _refresh_model_status(model)
    db.session.commit()


def _validate_model_payload(payload: Any, required: set[str]) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult(ok=False, error="Invalid JSON payload")

    missing = sorted(field for field in required if field not in payload)
    if missing:
        return ValidationResult(ok=False, error=f"Missing required fields: {', '.join(missing)}")

    data: dict[str, Any] = {}
    for key in ["name", "host", "backend", "model_name"]:
        if key in payload:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                return ValidationResult(ok=False, error=f"{key} must be a non-empty string")
            data[key] = value.strip()

    if "port" in payload:
        try:
            data["port"] = int(payload["port"])
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="port must be an integer")

    return ValidationResult(ok=True, data=data)


def _validate_agent_payload(payload: Any, required: set[str]) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult(ok=False, error="Invalid JSON payload")

    missing = sorted(field for field in required if field not in payload)
    if missing:
        return ValidationResult(ok=False, error=f"Missing required fields: {', '.join(missing)}")

    data: dict[str, Any] = {}

    if "name" in payload:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(ok=False, error="name must be a non-empty string")
        data["name"] = name.strip()

    if "model_id" in payload:
        try:
            model_id = int(payload.get("model_id"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="model_id must be an integer")

        if db.session.get(Model, model_id) is None:
            return ValidationResult(ok=False, error="model_id does not exist")
        data["model_id"] = model_id

    if "system_prompt" in payload:
        prompt = payload.get("system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ValidationResult(ok=False, error="system_prompt must be a non-empty string")
        data["system_prompt"] = prompt

    if "max_tokens" in payload:
        try:
            data["max_tokens"] = int(payload.get("max_tokens"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="max_tokens must be an integer")

    if "temperature" in payload:
        try:
            data["temperature"] = float(payload.get("temperature"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="temperature must be a number")

    return ValidationResult(ok=True, data=data)


@setup_bp.get("/models")
def get_models() -> Any:
    _refresh_all_model_statuses()
    models = Model.query.order_by(Model.id.asc()).all()
    return jsonify({"success": True, "data": [_model_to_dict(model) for model in models]})


@setup_bp.post("/models")
def create_model() -> Any:
    validation = _validate_model_payload(
        request.get_json(silent=True),
        required={"name", "host", "port", "backend", "model_name"},
    )
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error}), 400

    model = Model(**validation.data)
    db.session.add(model)
    try:
        db.session.flush()
        _refresh_model_status(model)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"success": False, "error": "Model name must be unique"}), 400

    return jsonify({"success": True, "data": _model_to_dict(model)}), 201


@setup_bp.put("/models/<int:model_id>")
def update_model(model_id: int) -> Any:
    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    validation = _validate_model_payload(
        request.get_json(silent=True),
        required=set(),
    )
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error}), 400

    for key, value in validation.data.items():
        setattr(model, key, value)

    try:
        _refresh_model_status(model)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"success": False, "error": "Model name must be unique"}), 400

    return jsonify({"success": True, "data": _model_to_dict(model)})


@setup_bp.delete("/models/<int:model_id>")
def delete_model(model_id: int) -> Any:
    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    db.session.delete(model)
    db.session.commit()
    return jsonify({"success": True, "data": {"message": "Model deleted"}})


@setup_bp.post("/models/probe")
def probe_models() -> Any:
    _refresh_all_model_statuses()
    models = Model.query.order_by(Model.id.asc()).all()
    return jsonify({"success": True, "data": [_model_to_dict(model) for model in models]})


@setup_bp.get("/agents")
def get_agents() -> Any:
    _refresh_all_model_statuses()
    agents = Agent.query.order_by(Agent.id.asc()).all()
    return jsonify({"success": True, "data": [_agent_to_dict(agent) for agent in agents]})


@setup_bp.post("/agents")
def create_agent() -> Any:
    validation = _validate_agent_payload(
        request.get_json(silent=True),
        required={"name", "model_id", "system_prompt", "max_tokens", "temperature"},
    )
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error}), 400

    agent = Agent(**validation.data)
    db.session.add(agent)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"success": False, "error": "Agent name must be unique"}), 400

    return jsonify({"success": True, "data": _agent_to_dict(agent)}), 201


@setup_bp.put("/agents/<int:agent_id>")
def update_agent(agent_id: int) -> Any:
    agent = db.session.get(Agent, agent_id)
    if agent is None:
        return jsonify({"success": False, "error": "Agent not found"}), 404

    validation = _validate_agent_payload(request.get_json(silent=True), required=set())
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error}), 400

    for key, value in validation.data.items():
        setattr(agent, key, value)

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"success": False, "error": "Agent name must be unique"}), 400

    return jsonify({"success": True, "data": _agent_to_dict(agent)})


@setup_bp.delete("/agents/<int:agent_id>")
def delete_agent(agent_id: int) -> Any:
    agent = db.session.get(Agent, agent_id)
    if agent is None:
        return jsonify({"success": False, "error": "Agent not found"}), 404

    db.session.delete(agent)
    db.session.commit()
    return jsonify({"success": True, "data": {"message": "Agent deleted"}})

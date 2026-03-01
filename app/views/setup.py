from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import Blueprint, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.connectors import Connector, ConnectorError, MLXConnector, OllamaConnector, TensorRTConnector
from app.connectors.detector import detect_backend
from app.connectors.probe import probe_backend
from app.extensions import db
from app.models import Agent, Model
from app.security import sanitize_text_input, validate_host
from app.services.model_warm import warm_model
from app.services.status_service import (
    build_agent_status_payload,
    build_model_status_payload,
    check_engine,
    compute_agent_status,
    compute_model_status,
    record_model_load,
    to_human,
)

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
        "engine": model.engine or model.backend,
        "model_name": model.model_name,
        "selected_model": model.selected_model,
        "status": model.status,
        "warm_status": model.warm_status,
        "last_warmed_at": model.last_warmed_at.isoformat() if model.last_warmed_at else None,
        "last_load_attempt_at": model.last_load_attempt_at.isoformat() if model.last_load_attempt_at else None,
        "last_load_message": model.last_load_message,
        "last_engine_check_at": model.last_engine_check_at.isoformat() if model.last_engine_check_at else None,
        "last_engine_message": model.last_engine_message,
    }


def _agent_to_dict(agent: Agent) -> dict[str, Any]:
    engine_state = {
        "reachable": bool(agent.model and agent.model.status == "green"),
    }
    model_state = {
        "load_state": "warm" if agent.model and agent.model.warm_status == "warm" else "cold",
    }
    aggregate_status = compute_agent_status(agent, model_state, engine_state)
    effective_status = {
        "ready": "green",
        "partially_ready": "yellow",
        "not_ready": "red",
        "disabled": "gray",
    }[aggregate_status]
    return {
        "id": agent.id,
        "name": agent.name,
        "model_id": agent.model_id,
        "model_name": agent.model.name if agent.model else None,
        "engine": (agent.model.engine if agent.model else None),
        "system_prompt": agent.system_prompt,
        "max_tokens": agent.max_tokens,
        "temperature": agent.temperature,
        "status": agent.status,
        "model_status": agent.model.status if agent.model else "red",
        "effective_status": effective_status,
        "aggregate_status": aggregate_status,
    }


def _connector_for_model(model: Model) -> Connector:
    backend = (model.engine or model.backend).lower()
    if backend == "ollama":
        return OllamaConnector(host=model.host, port=model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {model.backend}")


def _refresh_model_status(model: Model) -> None:
    from datetime import datetime

    checked_at = datetime.utcnow()
    model.last_engine_check_at = checked_at
    try:
        connector = _connector_for_model(model)
        connector.probe()
        models = connector.list_models()
        model.last_engine_message = "ok"
        engine_state = {"reachable": True}
        model.status = "green"
    except (ConnectorError, ValueError) as exc:
        models = []
        model.last_engine_message = str(exc) or "connection failed"
        engine_state = {"reachable": False}
        model.status = "red"

    model_state = compute_model_status(model, engine_state, models)
    model.warm_status = model_state["load_state"]


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
    for key in ["name", "host", "backend", "engine", "model_name", "selected_model"]:
        if key in payload:
            value = payload.get(key)
            try:
                data[key] = sanitize_text_input(value, key, max_length=255)
            except ValueError as exc:
                return ValidationResult(ok=False, error=str(exc))

    if "host" in data and not validate_host(data["host"]):
        return ValidationResult(ok=False, error="host contains invalid characters")

    if "port" in payload:
        try:
            data["port"] = int(payload["port"])
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="port must be an integer")
        if not 1 <= data["port"] <= 65535:
            return ValidationResult(ok=False, error="port must be between 1 and 65535")

    return ValidationResult(ok=True, data=data)




def _active_backend_engine() -> str | None:
    active = Model.query.filter_by(status="green").order_by(Model.id.asc()).first()
    if active is None:
        return None
    return (active.engine or active.backend).lower()

def _validate_agent_payload(payload: Any, required: set[str]) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult(ok=False, error="Invalid JSON payload")

    missing = sorted(field for field in required if field not in payload)
    if missing:
        return ValidationResult(ok=False, error=f"Missing required fields: {', '.join(missing)}")

    data: dict[str, Any] = {}

    if "name" in payload:
        name = payload.get("name")
        try:
            data["name"] = sanitize_text_input(name, "name", max_length=255)
        except ValueError as exc:
            return ValidationResult(ok=False, error=str(exc))

    if "model_id" in payload:
        try:
            model_id = int(payload.get("model_id"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="model_id must be an integer")

        model = db.session.get(Model, model_id)
        if model is None:
            return ValidationResult(ok=False, error="model_id does not exist")
        active_engine = _active_backend_engine()
        model_engine = (model.engine or model.backend).lower()
        if active_engine and model_engine != active_engine:
            return ValidationResult(ok=False, error="Engine mismatch")
        data["model_id"] = model_id

    if "system_prompt" in payload:
        prompt = payload.get("system_prompt")
        try:
            data["system_prompt"] = sanitize_text_input(prompt, "system_prompt", max_length=12000)
        except ValueError as exc:
            return ValidationResult(ok=False, error=str(exc))

    if "max_tokens" in payload:
        try:
            data["max_tokens"] = int(payload.get("max_tokens"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="max_tokens must be an integer")
        if data["max_tokens"] < 1:
            return ValidationResult(ok=False, error="max_tokens must be greater than 0")

    if "temperature" in payload:
        try:
            data["temperature"] = float(payload.get("temperature"))
        except (TypeError, ValueError):
            return ValidationResult(ok=False, error="temperature must be a number")
        if not 0 <= data["temperature"] <= 2:
            return ValidationResult(ok=False, error="temperature must be between 0 and 2")

    return ValidationResult(ok=True, data=data)


@setup_bp.get("/models")
def get_models() -> Any:
    try:
        _refresh_all_model_statuses()
        models = Model.query.order_by(Model.id.asc()).all()
        return jsonify({"success": True, "data": [_model_to_dict(model) for model in models]})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@setup_bp.post("/models")
def create_model() -> Any:
    validation = _validate_model_payload(
        request.get_json(silent=True),
        required={"name", "host", "port", "engine", "model_name"},
    )
    if not validation.ok:
        return jsonify({"success": False, "error": validation.error}), 400

    payload = dict(validation.data)
    payload["backend"] = payload.get("engine", payload.get("backend", "ollama"))
    model = Model(**payload)
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

    update_data = dict(validation.data)
    if "engine" in update_data and "backend" not in update_data:
        update_data["backend"] = update_data["engine"]
    for key, value in update_data.items():
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
    try:
        payload = request.get_json(force=True)
        host = sanitize_text_input(payload.get("host"), "host", max_length=255)
        if not validate_host(host):
            return jsonify({"success": False, "error": "host contains invalid characters"}), 400

        port = int(payload.get("port"))
        if not 1 <= port <= 65535:
            return jsonify({"success": False, "error": "port must be between 1 and 65535"}), 400

        engine = sanitize_text_input(payload.get("engine"), "engine", max_length=64).lower()
        models = probe_backend(host, port, engine)
        return jsonify({"success": True, "models": models, "data": {"models": models}})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400




@setup_bp.post("/models/warm")
def warm_model_endpoint() -> Any:
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")
    try:
        model_id = int(model_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "model_id must be an integer"}), 400

    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    status = warm_model(model)
    ok = status == "warm"
    message = "loaded ok" if ok else "failed to load"
    record_model_load(model, ok=ok, message=message)
    db.session.commit()
    return jsonify({"success": True, "status": status, "data": {"model_id": model.id, "warm_status": status, "message": message}})


@setup_bp.get("/models/status/<int:model_id>")
def get_model_status(model_id: int) -> Any:
    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    payload = build_model_status_payload(model)
    return jsonify({"success": True, "warm_status": payload["model"]["load_state"], "data": {"model_id": model.id, "warm_status": payload["model"]["load_state"]}})


@setup_bp.post("/models/test")
def test_model_backend() -> Any:
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

        host = sanitize_text_input(payload.get("host"), "host", max_length=255)
        if not validate_host(host):
            return jsonify({"success": False, "error": "host contains invalid characters"}), 400

        port = int(payload.get("port", 9001))
        if not 1 <= port <= 65535:
            return jsonify({"success": False, "error": "port must be between 1 and 65535"}), 400

        engine = payload.get("engine")
        if engine:
            normalized_engine = sanitize_text_input(engine, "engine", max_length=64).lower()
            models = probe_backend(host, port, normalized_engine)
            return jsonify({
                "success": True,
                "models": models,
                "data": {
                    "backend": normalized_engine,
                    "models": models,
                },
            })

        result = detect_backend(host, port)
        if not result:
            return jsonify({"success": False, "error": "Unable to detect backend"}), 400

        return jsonify({
            "success": True,
            "data": {
                "backend": result["backend"],
                "backend_version": result.get("version"),
                "models": result.get("models", []),
            },
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

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


@setup_bp.get("/v1/models/<int:model_id>/status")
def get_model_status_v1(model_id: int) -> Any:
    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404
    payload = build_model_status_payload(model)
    tooltip = (
        f"Inference engine reachable. Last checked {to_human(model.last_engine_check_at)}."
        if payload["engine"]["reachable"]
        else f"Inference engine unreachable at {payload['engine']['host']}. Last checked {to_human(model.last_engine_check_at)}. Click for diagnostics and retry."
    )
    payload["engine"]["tooltip"] = tooltip
    state = payload["model"]["load_state"]
    if state == "not_present":
        payload["model"]["tooltip"] = "Model files not found on host. Add model files or update model path."
    elif state == "cold":
        payload["model"]["tooltip"] = "Model present on disk but not loaded in engine. Click to load."
    else:
        payload["model"]["tooltip"] = f"Model loaded in inference engine (cached). Last loaded {to_human(model.last_load_attempt_at)}."
    return jsonify(payload)


@setup_bp.get("/v1/agents/<int:agent_id>/status")
def get_agent_status_v1(agent_id: int) -> Any:
    agent = db.session.get(Agent, agent_id)
    if agent is None:
        return jsonify({"success": False, "error": "Agent not found"}), 404
    payload = build_agent_status_payload(agent)
    tt = {
        "ready": "Agent ready to accept queries: engine reachable and model loaded.",
        "partially_ready": "Agent is configured but runtime unavailable (engine unreachable or model not loaded). Click for diagnostics.",
        "not_ready": "Agent cannot run (model missing or configuration error). Click to view error.",
        "disabled": "Agent is disabled.",
    }
    payload["agent"]["tooltip"] = tt[payload["agent"]["status"]]
    return jsonify(payload)


@setup_bp.get("/v1/status")
def get_status_v1() -> Any:
    model_id = request.args.get("model_id", type=int)
    agent_id = request.args.get("agent_id", type=int)
    if model_id:
        model = db.session.get(Model, model_id)
        if model is None:
            return jsonify({"success": False, "error": "Model not found"}), 404
        return jsonify(build_model_status_payload(model))
    if agent_id:
        agent = db.session.get(Agent, agent_id)
        if agent is None:
            return jsonify({"success": False, "error": "Agent not found"}), 404
        return jsonify(build_agent_status_payload(agent))
    return jsonify({"success": False, "error": "model_id or agent_id is required"}), 400


@setup_bp.post("/v1/engine/check")
def force_engine_check_v1() -> Any:
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id")
    try:
        model_id = int(model_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "model_id must be an integer"}), 400

    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    engine = check_engine(model)
    db.session.commit()
    return jsonify({"success": True, "engine": engine})


@setup_bp.post("/v1/models/<int:model_id>/reload")
def reload_model_v1(model_id: int) -> Any:
    model = db.session.get(Model, model_id)
    if model is None:
        return jsonify({"success": False, "error": "Model not found"}), 404

    status = warm_model(model)
    ok = status == "warm"
    message = "loaded ok" if ok else "failed to load"
    record_model_load(model, ok=ok, message=message)
    db.session.commit()
    return jsonify({"success": True, "model": {"id": model.id, "load_state": status, "message": message}})

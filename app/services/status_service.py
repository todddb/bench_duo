from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from app.connectors import ConnectorError, MLXConnector, OllamaConnector, TensorRTConnector
from app.models import Agent, Model

_STATUS_LOGS: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=50))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_human(value: datetime | None) -> str:
    if value is None:
        return "never"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _connector_for_model(model: Model):
    backend = (model.engine or model.backend).lower()
    if backend == "ollama":
        return OllamaConnector(host=model.host, port=model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm", "tensorrt-llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {model.backend}")


def _append_log(key: str, message: str, at: datetime | None = None) -> None:
    timestamp = at or _utcnow()
    _STATUS_LOGS[key].append(f"{to_iso(timestamp)} {message}")


def recent_logs_for_model(model: Model, limit: int = 5) -> list[str]:
    return list(_STATUS_LOGS.get(f"model:{model.id}", []))[-limit:]


def check_engine(model: Model) -> dict[str, Any]:
    checked_at = _utcnow()
    host = f"{model.host}:{model.port}"
    try:
        connector = _connector_for_model(model)
        connector.probe()
        message = "ok"
        model.last_engine_check_at = checked_at
        model.last_engine_message = message
        _append_log(f"model:{model.id}", f"engine check ok at {host}", checked_at)
        return {
            "reachable": True,
            "last_checked": to_iso(checked_at),
            "message": message,
            "host": host,
        }
    except (ConnectorError, ValueError) as exc:
        message = str(exc) or "connection failed"
        model.last_engine_check_at = checked_at
        model.last_engine_message = message
        _append_log(f"model:{model.id}", f"engine check failed at {host}: {message}", checked_at)
        return {
            "reachable": False,
            "last_checked": to_iso(checked_at),
            "message": message,
            "host": host,
        }


def get_engine_state(model: Model) -> dict[str, Any]:
    host = f"{model.host}:{model.port}"
    return {
        "reachable": model.status == "green",
        "last_checked": to_iso(model.last_engine_check_at),
        "message": model.last_engine_message or ("ok" if model.status == "green" else "status unknown"),
        "host": host,
    }


def compute_model_status(model_record: Model, engine_state: dict[str, Any], loaded_models: list[str] | None = None) -> dict[str, Any]:
    if model_record.warm_status == "error":
        exists_on_disk = False
    else:
        exists_on_disk = True

    if not exists_on_disk:
        return {
            "exists_on_disk": False,
            "loaded_in_engine": False,
            "load_state": "not_present",
        }

    if engine_state.get("reachable") and model_record.model_name in (loaded_models or []):
        return {
            "exists_on_disk": True,
            "loaded_in_engine": True,
            "load_state": "warm",
        }

    if model_record.warm_status == "warm":
        return {
            "exists_on_disk": True,
            "loaded_in_engine": bool(engine_state.get("reachable")),
            "load_state": "warm" if engine_state.get("reachable") else "cold",
        }

    return {
        "exists_on_disk": True,
        "loaded_in_engine": False,
        "load_state": "cold",
    }


def compute_agent_status(agent_record: Agent, model_status: dict[str, Any], engine_state: dict[str, Any]) -> str:
    enabled = agent_record.status != "disabled"
    if not enabled:
        return "disabled"
    if model_status["load_state"] == "warm" and engine_state["reachable"]:
        return "ready"
    if model_status["load_state"] in ("warm", "cold") and not engine_state["reachable"]:
        return "partially_ready"
    return "not_ready"


def _fetch_loaded_models(model: Model) -> list[str]:
    try:
        connector = _connector_for_model(model)
        return connector.list_models()
    except (ConnectorError, ValueError):
        return []


def build_model_status_payload(model: Model, force_engine_check: bool = False) -> dict[str, Any]:
    engine_state = check_engine(model) if force_engine_check else get_engine_state(model)
    loaded_models = _fetch_loaded_models(model) if engine_state.get("reachable") else []
    model_state = compute_model_status(model, engine_state, loaded_models)
    return {
        "engine": engine_state,
        "model": {
            **model_state,
            "last_load_attempt": to_iso(model.last_load_attempt_at),
            "last_load_message": model.last_load_message,
        },
        "logs": {"recent": recent_logs_for_model(model, 5)},
    }


def build_agent_status_payload(agent: Agent, force_engine_check: bool = False) -> dict[str, Any]:
    model_payload = build_model_status_payload(agent.model, force_engine_check=force_engine_check)
    engine_state = model_payload["engine"]
    model_state = model_payload["model"]
    status = compute_agent_status(agent, model_state, engine_state)
    return {
        **model_payload,
        "model_id": agent.model_id,
        "agent": {
            "enabled": agent.status != "disabled",
            "status": status,
            "diagnostics": {
                "last_agent_init": to_iso(agent.updated_at),
                "last_agent_message": f"agent status={status}",
            },
        },
    }


def record_model_load(model: Model, ok: bool, message: str) -> None:
    at = _utcnow()
    model.last_load_attempt_at = at
    model.last_load_message = message
    _append_log(f"model:{model.id}", f"model reload {'ok' if ok else 'failed'}: {message}", at)

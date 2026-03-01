from __future__ import annotations

from typing import Any

import requests


def probe_backend(host: str, port: int, engine: str) -> list[str]:
    base = f"http://{host}:{port}"
    normalized = engine.lower()

    if normalized == "ollama":
        response = requests.get(f"{base}/api/tags", timeout=3)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [model.get("name", "") for model in data.get("models", []) if model.get("name")]

    if normalized == "mlx":
        response = requests.get(f"{base}/v1/models", timeout=3)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [model.get("id", "") for model in data.get("data", []) if model.get("id")]

    if normalized in {"tensorrt-llm", "tensorrt", "tensorrt_llm"}:
        response = requests.get(f"{base}/models", timeout=3)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [model for model in data.get("models", []) if isinstance(model, str)]

    raise ValueError("Unknown engine")

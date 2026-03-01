from __future__ import annotations

from typing import Any

import requests


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def detect_backend(host: str, port: int, timeout: float = 3.0) -> dict[str, Any] | None:
    base_url = f"http://{host}:{port}"

    # Try Ollama
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            data = _safe_json(response)
            models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
            return {
                "backend": "ollama",
                "version": data.get("version"),
                "models": models,
            }
    except requests.RequestException:
        pass

    # Try MLX/OpenAI-compatible endpoint
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if response.status_code == 200:
            data = _safe_json(response)
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
            return {
                "backend": "mlx",
                "version": data.get("version"),
                "models": models,
            }
    except requests.RequestException:
        pass

    # Try TensorRT-LLM
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        if response.status_code == 200:
            data = _safe_json(response)
            models_raw = data.get("models", [])
            models = [m for m in models_raw if isinstance(m, str)]
            return {
                "backend": "tensorrt-llm",
                "version": data.get("version"),
                "models": models,
            }
    except requests.RequestException:
        pass

    return None

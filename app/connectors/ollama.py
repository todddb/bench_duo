from __future__ import annotations

from typing import Any

import requests

from .base import Connector, ConnectorError


class OllamaConnector(Connector):
    def __init__(self, host: str = "localhost", port: int = 11434, timeout: float = 10.0) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def probe(self) -> dict[str, Any]:
        endpoints = ("/api/health", "/version")
        last_error: Exception | None = None

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                response.raise_for_status()
                data = response.json() if response.content else {}
                return {"ok": True, "endpoint": endpoint, "data": data}
            except (requests.RequestException, ValueError) as exc:
                last_error = exc

        raise ConnectorError(f"Ollama probe failed: {last_error}")

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ConnectorError(f"Failed to list Ollama models: {exc}") from exc

        models = payload.get("models", [])
        return [model.get("name", "") for model in models if model.get("name")]

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        model_name = settings.get("model") or settings.get("model_name")
        if not model_name:
            raise ConnectorError("Ollama chat requires `model` or `model_name` in settings")

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.get("temperature"),
                "num_predict": settings.get("max_tokens"),
            },
        }
        payload["options"] = {k: v for k, v in payload["options"].items() if v is not None}

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=settings.get("timeout", self.timeout),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ConnectorError(f"Ollama chat failed: {exc}") from exc

        content = (data.get("message") or {}).get("content")
        if not content:
            raise ConnectorError("Ollama chat response missing message content")
        return content

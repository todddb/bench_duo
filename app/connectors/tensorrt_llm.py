from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import Connector, ConnectorError


class TensorRTConnector(Connector):
    def __init__(self, default_models: list[str] | None = None) -> None:
        self.default_models = default_models or ["local-tensorrt-model"]

    def probe(self) -> dict[str, Any]:
        try:
            import_module("tensorrt_llm")
        except ImportError as exc:
            raise ConnectorError("tensorrt_llm is not installed") from exc
        return {"ok": True, "backend": "tensorrt_llm"}

    def list_models(self) -> list[str]:
        return self.default_models

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        model_name = settings.get("model") or settings.get("model_name")
        if not model_name:
            raise ConnectorError("TensorRT chat requires `model` or `model_name` in settings")

        try:
            trt_module = import_module("tensorrt_llm")
            llm = trt_module.LLM(model=model_name)
            sampling_params = trt_module.SamplingParams(
                temperature=settings.get("temperature", 0.2),
                top_p=settings.get("top_p", 0.95),
                max_tokens=settings.get("max_tokens", 256),
            )
            outputs = llm.chat(messages, sampling_params=sampling_params)
        except ImportError as exc:
            raise ConnectorError("tensorrt_llm is not installed") from exc
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(f"TensorRT generation failed: {exc}") from exc

        try:
            content = outputs[0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ConnectorError("TensorRT response missing assistant message content") from exc

        if not content:
            raise ConnectorError("TensorRT generation returned empty output")
        return content

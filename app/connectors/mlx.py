from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import Connector, ConnectorError


class MLXConnector(Connector):
    def __init__(self, default_models: list[str] | None = None) -> None:
        self.default_models = default_models or ["mlx-community/Llama-3.2-3B-Instruct-4bit"]

    def probe(self) -> dict[str, Any]:
        try:
            import_module("mlx_lm")
        except ImportError as exc:
            raise ConnectorError("mlx_lm is not installed") from exc
        return {"ok": True, "backend": "mlx_lm"}

    def list_models(self) -> list[str]:
        return self.default_models

    def chat(self, messages: list[dict[str, str]], settings: dict[str, Any]) -> str:
        model_name = settings.get("model") or settings.get("model_name")
        if not model_name:
            raise ConnectorError("MLX chat requires `model` or `model_name` in settings")

        try:
            mlx_lm = import_module("mlx_lm")
            model, tokenizer = mlx_lm.load(model_name)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text = mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                temp=settings.get("temperature", 0.2),
                max_tokens=settings.get("max_tokens", 256),
            )
        except ImportError as exc:
            raise ConnectorError("mlx_lm is not installed") from exc
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(f"MLX generation failed: {exc}") from exc

        if not text:
            raise ConnectorError("MLX generation returned empty output")
        return text

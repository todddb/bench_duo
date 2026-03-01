from __future__ import annotations

from datetime import datetime

import requests

from app.extensions import db
from app.models import Model


SUPPORTED_WARM_STATUSES = {"cold", "loading", "warm", "error"}


def warm_model(model: Model, timeout: float = 10.0) -> str:
    model.warm_status = "loading"
    db.session.commit()

    try:
        base = f"http://{model.host}:{model.port}"
        target_model = model.selected_model or model.model_name

        if model.engine == "ollama":
            response = requests.post(
                f"{base}/api/generate",
                json={
                    "model": target_model,
                    "prompt": "",
                    "stream": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()
        elif model.engine == "mlx":
            response = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": target_model,
                    "messages": [{"role": "user", "content": "."}],
                    "max_tokens": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
        elif model.engine in {"tensorrt", "tensorrt_llm", "tensorrt-llm"}:
            response = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": target_model,
                    "messages": [{"role": "user", "content": "."}],
                    "max_tokens": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
        else:
            raise ValueError(f"Unsupported engine for warm loading: {model.engine}")

        model.warm_status = "warm"
        model.last_warmed_at = datetime.utcnow()
    except Exception:
        model.warm_status = "error"

    db.session.commit()
    return model.warm_status

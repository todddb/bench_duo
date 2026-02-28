from flask import Blueprint

health_bp = Blueprint("health", __name__, url_prefix="/health")


@health_bp.get("/")
def health() -> dict[str, str]:
    return {"status": "healthy"}

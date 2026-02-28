from flask import Flask

from .config import Config
from .extensions import db, socketio
from .views.health import health_bp


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    app.register_blueprint(health_bp)

    @app.get("/")
    def index() -> dict[str, str]:
        return {"name": "bench_duo", "status": "ok"}

    return app

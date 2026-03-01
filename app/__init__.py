from flask import Flask, jsonify

from .config import Config
from .extensions import db, socketio
from . import models  # noqa: F401
from .views.health import health_bp
from .views.setup import setup_bp
from .views.chat import chat_bp, init_chat_worker
from .views.evaluate import evaluate_bp
from .views.batch import batch_bp, init_batch_worker
from .views.pages import pages_bp


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    app.register_blueprint(health_bp)
    app.register_blueprint(setup_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(evaluate_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(pages_bp)

    init_chat_worker(app)
    init_batch_worker(app)

    @app.errorhandler(404)
    def not_found(_: Exception):
        return jsonify({"success": False, "error": "Not Found"}), 404

    @app.errorhandler(500)
    def internal_error(_: Exception):
        return jsonify({"success": False, "error": "Internal Server Error"}), 500

    return app

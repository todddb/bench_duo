import os

import eventlet

eventlet.monkey_patch()

from app import create_app
from app.extensions import socketio

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting Bench Duo on http://{host}:{port}")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )

import sqlite3
from pathlib import Path

from app import create_app
from app.extensions import db
from scripts.init_db import _apply_sqlite_compat_migrations


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


def test_apply_sqlite_compat_migrations_adds_missing_model_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE models (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                name VARCHAR(255) NOT NULL UNIQUE,
                host VARCHAR(255) NOT NULL,
                port INTEGER NOT NULL,
                backend VARCHAR(64) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                status VARCHAR(32) NOT NULL
            )
            """
        )

    class FileConfig(TestConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"

    app = create_app(FileConfig)
    with app.app_context():
        _apply_sqlite_compat_migrations()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]: row[2]
            for row in connection.execute("PRAGMA table_info(models)")
        }

    assert "engine" in columns
    assert "selected_model" in columns
    db.session.remove()

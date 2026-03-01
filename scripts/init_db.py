import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.extensions import db


def _sqlite_tables(connection) -> set[str]:
    rows = connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in rows}


def _sqlite_columns(connection, table_name: str) -> set[str]:
    rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})")
    return {row[1] for row in rows}


def _apply_sqlite_compat_migrations() -> None:
    """Backfill columns for older local SQLite databases."""
    column_migrations = {
        "models": [
            ("engine", "TEXT NOT NULL DEFAULT 'ollama'"),
            ("selected_model", "TEXT"),
            ("warm_status", "TEXT NOT NULL DEFAULT 'cold'"),
            ("last_warmed_at", "DATETIME"),
            ("last_load_attempt_at", "DATETIME"),
            ("last_load_message", "TEXT"),
            ("last_engine_check_at", "DATETIME"),
            ("last_engine_message", "TEXT"),
        ],
        "conversations": [
            ("random_seed", "INTEGER"),
            ("status", "TEXT NOT NULL DEFAULT 'pending'"),
        ],
        "messages": [
            ("raw_response", "JSON"),
        ],
        "batch_jobs": [
            ("completed_runs", "INTEGER NOT NULL DEFAULT 0"),
            ("cancel_requested", "BOOLEAN NOT NULL DEFAULT 0"),
            ("summary", "JSON"),
        ],
        "evaluation_jobs": [
            ("report", "JSON"),
            ("status", "TEXT NOT NULL DEFAULT 'pending'"),
        ],
    }

    with db.engine.begin() as connection:
        existing_tables = _sqlite_tables(connection)
        for table_name, columns in column_migrations.items():
            if table_name not in existing_tables:
                continue
            existing_columns = _sqlite_columns(connection, table_name)
            for column_name, column_def in columns:
                if column_name in existing_columns:
                    continue
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
                )
                print(f"Added missing column: {table_name}.{column_name}")


def main() -> None:
    app = create_app()
    with app.app_context():
        db.create_all()
        if db.engine.dialect.name == "sqlite":
            _apply_sqlite_compat_migrations()
        print("Database initialized.")


if __name__ == "__main__":
    main()

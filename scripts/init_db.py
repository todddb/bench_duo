from app import create_app
from app.extensions import db


def _sqlite_columns(connection, table_name: str) -> set[str]:
    rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})")
    return {row[1] for row in rows}


def _apply_sqlite_compat_migrations() -> None:
    """Backfill columns for older local SQLite databases."""
    column_migrations = {
        "models": [
            ("engine", "TEXT NOT NULL DEFAULT 'ollama'"),
            ("selected_model", "TEXT"),
        ],
    }

    with db.engine.begin() as connection:
        for table_name, columns in column_migrations.items():
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

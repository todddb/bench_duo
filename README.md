# bench_duo

Flask-based AI Duo application skeleton for qualitative LLM testing.

## Quick start

```bash
./scripts/setup.sh
source .venv/bin/activate
python run.py
```

## Installer behavior

`scripts/setup.sh` now performs:

1. OS detection (`$OSTYPE`) for Linux/macOS package installation.
2. System dependency installation (`apt-get` on Linux, `brew` on macOS).
3. Virtual environment creation and `pip install -r requirements.txt`.
4. SQLite database initialization (`scripts/init_db.py`).
5. Smoke tests (`scripts/smoke_test.sh`) to verify toolchain and app startup imports.

Optional flags:

- `SKIP_SYSTEM_DEPS=1 ./scripts/setup.sh` to skip `apt-get`/`brew`.
- `RUN_SMOKE_TESTS=0 ./scripts/setup.sh` to skip post-install smoke tests.

## Smoke-test commands

Run manually after setup:

```bash
./scripts/smoke_test.sh
```

The smoke test validates:

- `sqlite3 --version`
- `python -c "import flask, sqlalchemy"`
- `python scripts/init_db.py`
- `python -c "from app import create_app; create_app()"`

## Export endpoints

- Conversation export: `GET /api/conversations/<id>/export`
  - JSON (default): `?format=json`
  - CSV: `?format=csv`
- Batch export: `GET /api/batch_jobs/<id>/export`
  - JSON (default): `?format=json`
  - CSV: `?format=csv`

The Chat and Batch pages include export buttons for JSON/CSV downloads.

## Retention & purge

- API purge endpoint for batch jobs: `POST /api/batch_jobs/purge` with optional JSON body:
  - `{"older_than_days": 30}`
- Management script for scheduled/manual cleanup:

```bash
python scripts/purge_old_jobs.py --days 30
```

## Security notes

- Use HTTPS in production to protect prompts, model settings, and generated content in transit.
- Add authentication/authorization (for example, API tokens) before exposing write endpoints.
- Do **not** expose underlying LLM backends/connectors directly to the public internet.
- Prompt and text inputs are validated/sanitized before persistence; ORM-based parameterized queries are used to reduce SQL injection risk.
- Host and port inputs are validated to reduce URL/host injection risks in model connector configuration.

## Project layout

- `app/` Flask app package (config, models, views, extensions)
- `scripts/` setup and DB bootstrap scripts
- `instance/` runtime SQLite database files

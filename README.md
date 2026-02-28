# bench_duo

Flask-based AI Duo application skeleton for qualitative LLM testing.

## Quick start

```bash
./scripts/setup.sh
source .venv/bin/activate
python run.py
```

## Project layout

- `app/` Flask app package (config, models, views, extensions)
- `scripts/` setup and DB bootstrap scripts
- `instance/` runtime SQLite database files

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run ./scripts/setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

run_check() {
  local description="$1"
  shift
  echo "[check] ${description}"
  "$@"
}

run_check "sqlite3 availability" sqlite3 --version
run_check "python imports" python -c "import flask, sqlalchemy; print('flask/sqlalchemy import ok')"
run_check "database initialization script" python "${ROOT_DIR}/scripts/init_db.py"
run_check "Flask app factory" python -c "from app import create_app; app=create_app(); print('app created:', app.name)"

echo "All smoke tests passed."

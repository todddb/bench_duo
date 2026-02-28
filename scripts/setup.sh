#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-0}"
RUN_SMOKE_TESTS="${RUN_SMOKE_TESTS:-1}"

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_system_deps() {
  if [[ "${SKIP_SYSTEM_DEPS}" == "1" ]]; then
    echo "Skipping system dependency installation (SKIP_SYSTEM_DEPS=1)."
    return
  fi

  if [[ "${OSTYPE:-}" == darwin* ]]; then
    echo "Detected macOS. Installing system dependencies with Homebrew..."
    command -v brew >/dev/null 2>&1 || {
      echo "Homebrew is required but not found. Please install Homebrew first." >&2
      exit 1
    }
    brew update
    brew install sqlite openssl readline pkg-config
  else
    echo "Detected Linux. Installing system dependencies with apt-get..."
    command -v apt-get >/dev/null 2>&1 || {
      echo "apt-get not found. Please install dependencies manually." >&2
      exit 1
    }
    run_privileged apt-get update
    run_privileged apt-get install -y sqlite3 build-essential libssl-dev libffi-dev python3-dev python3-venv
  fi
}

create_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip
  pip install -r "${ROOT_DIR}/requirements.txt"
}

init_db() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python "${ROOT_DIR}/scripts/init_db.py"
}

main() {
  install_system_deps
  create_venv
  init_db

  if [[ "${RUN_SMOKE_TESTS}" == "1" ]]; then
    "${ROOT_DIR}/scripts/smoke_test.sh"
  fi

  echo ""
  echo "Setup complete."
  echo "Activate virtualenv: source ${VENV_DIR}/bin/activate"
  echo "Run app: python ${ROOT_DIR}/run.py"
}

main "$@"

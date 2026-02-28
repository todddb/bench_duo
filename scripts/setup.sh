#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

install_system_deps() {
  local os_name
  os_name="$(uname -s)"

  if [[ "${os_name}" == "Darwin" ]]; then
    echo "Detected macOS. Installing system dependencies with Homebrew..."
    command -v brew >/dev/null 2>&1 || {
      echo "Homebrew is required but not found. Please install Homebrew first." >&2
      exit 1
    }
    brew install sqlite openssl readline pkg-config
  else
    echo "Detected Linux. Installing system dependencies with apt-get..."
    command -v apt-get >/dev/null 2>&1 || {
      echo "apt-get not found. Please install dependencies manually." >&2
      exit 1
    }
    sudo apt-get update
    sudo apt-get install -y sqlite3 build-essential libssl-dev libffi-dev python3-dev python3-venv
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

  echo "\nSetup complete."
  echo "Activate virtualenv: source ${VENV_DIR}/bin/activate"
  echo "Run app: python ${ROOT_DIR}/run.py"
}

main "$@"

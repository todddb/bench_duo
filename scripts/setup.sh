#!/usr/bin/env bash
set -e

echo "===== AI Duo Setup ====="

# Detect OS
OS_TYPE="$(uname)"
echo "Detected OS: $OS_TYPE"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Installing system dependencies via Homebrew..."
    brew install sqlite openssl readline pkgconf || true
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Installing system dependencies via apt..."
    sudo apt-get update
    sudo apt-get install -y sqlite3 build-essential libssl-dev libffi-dev python3.11 python3.11-venv
else
    echo "Unsupported OS"
    exit 1
fi

# Prefer python3.11
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
else
    PYTHON_BIN=python3
fi

PY_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PY_VERSION"

# Reject unsupported versions
$PYTHON_BIN - <<EOF2
import sys
if sys.version_info >= (3,14):
    print("ERROR: Python 3.14+ is not supported yet.")
    sys.exit(1)
EOF2

# Create venv
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

# Avoid building Rust wheels
export PIP_ONLY_BINARY=:all:

pip install --upgrade pip

pip install -r requirements.txt

echo "Initializing database..."
python scripts/init_db.py

echo ""
echo "===== Setup Complete ====="
echo "Activate with: source .venv/bin/activate"
echo "Start app with: ./scripts/duo start"
echo "Stop app with: ./scripts/duo stop"
echo "Check logs: tail -n 100 ./logs/bench_duo.log"

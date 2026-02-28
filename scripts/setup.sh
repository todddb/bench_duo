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
python - <<EOF2
from app import create_app
from app.extensions import db
app = create_app()
with app.app_context():
    db.create_all()
print("Database initialized.")
EOF2

echo ""
echo "===== Setup Complete ====="
echo "Activate with: source .venv/bin/activate"
echo "Run with: python run.py"

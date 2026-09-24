#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=psdenv
VENV_PY="$VENV_DIR/bin/python"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) VENV_PY="$VENV_DIR/Scripts/python.exe" ;; esac

# 1) Create the virtual environment (PyTorch needs Python 3.9 - 3.12).
if [ ! -x "$VENV_PY" ]; then
    echo "Creating virtual environment..."
    BASE_PY=""
    for candidate in python3.11 python3.12 python3.10 python3.9 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
            'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
            BASE_PY="$candidate"
            break
        fi
    done
    if [ -z "$BASE_PY" ]; then
        echo "Python 3.9 - 3.12 is required." >&2
        exit 1
    fi
    "$BASE_PY" -m venv "$VENV_DIR"
fi

# 2) Install requirements when requirements.txt changed since the last install.
if ! cmp -s requirements.txt "$VENV_DIR/.requirements"; then
    echo "Installing requirements..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
    cp requirements.txt "$VENV_DIR/.requirements"
fi

# 3) Download any missing models.
"$VENV_PY" main.py --download-models

# 4) Run. With no arguments this opens the web UI.
exec "$VENV_PY" main.py "$@"

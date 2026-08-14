#!/usr/bin/env bash
# Launch Guitar Trainer from the project's virtualenv.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [[ ! -x .venv/bin/python ]]; then
    echo "No virtualenv found. Setting one up..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e .
fi

# Qt6's xcb platform plugin needs this at runtime; without it the app aborts with
# "Could not load the Qt platform plugin xcb" instead of showing a window.
if command -v dpkg-query >/dev/null && ! dpkg-query -W -f='${Status}' libxcb-cursor0 2>/dev/null | grep -q "install ok installed"; then
    echo "Missing system package libxcb-cursor0 (required for the Qt display)."
    if command -v sudo >/dev/null; then
        sudo apt-get install -y libxcb-cursor0
    else
        echo "Install it manually: apt-get install -y libxcb-cursor0" >&2
        exit 1
    fi
fi

exec .venv/bin/python -m guitar_trainer "$@"

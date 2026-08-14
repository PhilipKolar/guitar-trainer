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

exec .venv/bin/python -m guitar_trainer "$@"

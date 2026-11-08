#!/usr/bin/env bash
# Quick launcher for Pashto Dual-Dialect Interactive Chat
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="$REPO_DIR/Pashto-LLM-V2/unsloth_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" scripts/chat_interactive.py "$@"


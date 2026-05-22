#!/usr/bin/env bash
set -euo pipefail

APP_CMD=(python -m streamlit run frontend/app.py)
FALLBACK_VENV="/home/youssef/.cache/pypoetry/virtualenvs/ml-assistant-rtt_Gyjj-py3.12/bin/activate"

if command -v poetry >/dev/null 2>&1 && poetry run python -c "import streamlit" >/dev/null 2>&1; then
  exec poetry run "${APP_CMD[@]}"
fi

if [[ -f "$FALLBACK_VENV" ]]; then
  # shellcheck disable=SC1090
  source "$FALLBACK_VENV"
  exec "${APP_CMD[@]}"
fi

echo "Poetry environment check failed and fallback virtualenv was not found:"
echo "  $FALLBACK_VENV"
exit 1

#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd "$(dirname "$0")" && pwd)"
environment_dir="$app_dir/.inspection-ui-venv"

cd "$app_dir"
if [[ ! -x "$environment_dir/bin/python" ]]; then
    python3 -m venv "$environment_dir"
    "$environment_dir/bin/python" -m pip install --upgrade pip
    "$environment_dir/bin/python" -m pip install -r requirements.txt
fi
"$environment_dir/bin/python" -m streamlit run streamlit_app.py

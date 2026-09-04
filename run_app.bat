@echo off
cd /d "%~dp0"
if not exist .inspection-ui-venv\Scripts\python.exe (
    python -m venv .inspection-ui-venv
    .inspection-ui-venv\Scripts\python.exe -m pip install --upgrade pip
    .inspection-ui-venv\Scripts\python.exe -m pip install -r requirements.txt
)
.inspection-ui-venv\Scripts\python.exe -m streamlit run streamlit_app.py

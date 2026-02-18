@echo off
REM LLM Evaluation Dashboard Launcher
cd /d "%~dp0"
echo.
echo 🚀 Starting LLM Evaluation Dashboard...
echo.
.\.venv\Scripts\python.exe app.py
if errorlevel 1 (
    echo.
    echo ❌ Error starting app. Check the messages above.
    echo.
    pause
)


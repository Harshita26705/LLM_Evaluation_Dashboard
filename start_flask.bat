@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ====================================
echo   LLM Evaluation Dashboard - Flask
echo ====================================
echo.
echo Starting Flask server...
echo.

if exist ".\.venv\Scripts\python.exe" (
	.\.venv\Scripts\python.exe flask_app.py
) else (
	python flask_app.py
)

pause

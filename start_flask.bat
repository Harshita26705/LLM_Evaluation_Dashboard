@echo off
setlocal enabledelayedexpansion
cd /d "C:\Users\HarshitaSuri\OneDrive - CG Infinity\Desktop\LLM_Dashboard"

echo.
echo ====================================
echo   LLM Evaluation Dashboard - Flask
echo ====================================
echo.
echo Starting Flask server...
echo.

.\.venv\Scripts\python.exe flask_app.py

pause

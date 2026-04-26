@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo Starting LLM Evaluation Dashboard...
if exist ".\.venv\Scripts\python.exe" (
	.\.venv\Scripts\python.exe app.py
) else (
	python app.py
)
pause

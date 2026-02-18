@echo off
cd /d "c:\Users\HarshitaSuri\OneDrive - CG Infinity\Desktop\LLM_Dashboard"
echo Installing spacy model...
.\.venv\Scripts\python.exe init_models.py
echo.
echo Setup complete! Run the app with:
echo   .\.venv\Scripts\python.exe app.py
pause

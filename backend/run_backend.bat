@echo off
echo Starting ML Energy Prediction API Backend...
echo ============================================
echo.
echo Backend will run on: http://localhost:8000
echo API Docs available at: http://localhost:8000/docs
echo.
cd /d "%~dp0"
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

@echo off
echo ============================================
echo  AI Product Strategy Assistant
echo ============================================

REM Check .env file
if not exist backend\.env (
    echo [WARN] backend\.env not found.
    echo Please copy .env.example to backend\.env and add your GROQ_API_KEY.
    echo.
)

REM Start FastAPI backend in a new window
echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI Backend" cmd /k "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for the backend to start
timeout /t 4 /nobreak >nul

REM Start Streamlit frontend
echo [2/2] Starting Streamlit frontend on http://localhost:8501 ...
start "Streamlit Frontend" cmd /k "cd frontend && streamlit run app.py --server.port 8501"

echo.
echo Both services are starting. Opening browser...
timeout /t 3 /nobreak >nul
start http://localhost:8501

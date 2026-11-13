@echo off
cd /d "D:\RAG-Chatbot"

echo Starting RAG Chatbot...

:: Start FastAPI Backend
start "FastAPI Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --reload"

:: Wait 5 seconds
timeout /t 5 /nobreak >nul

:: Start Streamlit Frontend
start "Streamlit Frontend" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py"

echo.
echo RAG Chatbot started!
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:8501

pause
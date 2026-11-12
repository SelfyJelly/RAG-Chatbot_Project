@echo off
cd /d "%~dp0"

echo Starting RAG Chatbot...

start "FastAPI Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --reload"

timeout /t 5 /nobreak >nul

start "Streamlit Frontend" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py"

echo RAG Chatbot started!
@echo off
chcp 65001 >nul 2>&1
cls
echo.
echo +============================================================================+
echo ^|            [FIRE] Starting FORGE-Guard [FIRE]                              ^|
echo +============================================================================+
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo [INFO] Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Check if dependencies are installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Dependencies not fully installed. Running setup...
    pip install -r requirements.txt --quiet
)

echo [INFO] Launching FORGE-Guard...
echo.
echo   [*] API Server:    http://localhost:8000
echo   [*] Dashboard:     http://localhost:8501
echo   [*] API Docs:      http://localhost:8000/docs
echo   [*] Admin Panel:   http://localhost:8501 (Settings tab)
echo.
echo   Press Ctrl+C to stop
echo.

:: Start the application
python main.py

pause

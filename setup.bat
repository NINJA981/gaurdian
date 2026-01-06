@echo off
chcp 65001 >nul 2>&1
cls
echo.
echo +============================================================================+
echo ^|                                                                            ^|
echo ^|            [FIRE] FORGE-Guard Setup Wizard [FIRE]                          ^|
echo ^|                 Elderly Safety Monitoring System                           ^|
echo ^|                                                                            ^|
echo +============================================================================+
echo.
echo [INFO] Checking system requirements...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo [INFO] Please install Python 3.10 or 3.11 from https://python.org
    echo [INFO] Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] Found Python %PYVER%

:: Check Python version compatibility
python -c "import sys; exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Python %PYVER% may have compatibility issues.
    echo [INFO] Recommended: Python 3.10 or 3.11 for best MediaPipe support.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo.
echo [STEP 1/4] Creating virtual environment...
echo.

:: Create virtual environment
if exist ".venv" (
    echo [INFO] Virtual environment already exists, recreating...
    rmdir /s /q .venv 2>nul
)
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment created

echo.
echo [STEP 2/4] Activating virtual environment...
echo.
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated

echo.
echo [STEP 3/4] Upgrading pip and installing dependencies...
echo [INFO] This may take a few minutes...
echo.

python -m pip install --upgrade pip setuptools wheel --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Pip upgrade had issues, continuing...
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install some dependencies!
    echo [INFO] Trying with --no-cache-dir...
    pip install -r requirements.txt --no-cache-dir
)

echo.
echo [STEP 4/4] Setting up environment configuration...
echo.

:: Create .env from example if not exists
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Created .env configuration file
    ) else (
        echo [INFO] No .env.example found, creating default .env...
        (
            echo # FORGE-Guard Configuration
            echo # ==========================
            echo.
            echo # Server Settings
            echo API_HOST=0.0.0.0
            echo API_PORT=8000
            echo STREAMLIT_PORT=8501
            echo.
            echo # Detection Settings
            echo FALL_DETECTION_ENABLED=true
            echo GESTURE_DETECTION_ENABLED=true
            echo MEDICINE_MONITORING_ENABLED=true
            echo OBJECT_DETECTION_ENABLED=true
            echo.
            echo # Alert Settings
            echo ALERT_COOLDOWN_SECONDS=30
            echo.
            echo # Twilio Settings ^(Optional - for SMS/Call alerts^)
            echo # TWILIO_ACCOUNT_SID=your_account_sid
            echo # TWILIO_AUTH_TOKEN=your_auth_token
            echo # TWILIO_PHONE_NUMBER=+1234567890
            echo # EMERGENCY_CONTACT=+1234567890
        ) > .env
        echo [OK] Created default .env configuration file
    )
) else (
    echo [OK] .env file already exists
)

:: Create logs directory
if not exist "logs" mkdir logs

echo.
echo +============================================================================+
echo ^|                                                                            ^|
echo ^|                    [SUCCESS] SETUP COMPLETE!                               ^|
echo ^|                                                                            ^|
echo +============================================================================+
echo ^|                                                                            ^|
echo ^|   To start FORGE-Guard, run:                                               ^|
echo ^|                                                                            ^|
echo ^|      run.bat                                                               ^|
echo ^|                                                                            ^|
echo ^|   Or manually:                                                             ^|
echo ^|      .venv\Scripts\activate                                                ^|
echo ^|      python main.py                                                        ^|
echo ^|                                                                            ^|
echo +============================================================================+
echo.
pause

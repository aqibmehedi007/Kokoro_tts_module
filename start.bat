@echo off
echo 🎤 QuteVoice TTS - Windows Startup
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python from https://python.org
    echo 📋 Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python detected
echo 🚀 Starting QuteVoice TTS Application...
echo ====================================

REM Run the Python application
python start.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ❌ Application failed to start
    pause
)

@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ================================
echo KSL Sign Translator Dashboard
echo ================================
echo.

set PYTHON_CMD=py

py --version > nul 2>&1
if errorlevel 1 (
    set PYTHON_CMD=python
)

%PYTHON_CMD% --version > nul 2>&1
if errorlevel 1 (
    echo Python was not found.
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements.txt

echo.
echo Starting dashboard...
echo Browser URL: http://localhost:8501
echo To stop, press Ctrl + C in this window.
echo.

%PYTHON_CMD% -m streamlit run dashboard.py

pause

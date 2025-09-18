@echo off
setlocal

echo ­⚙️  Creating virtual environment...

REM --- Check Python version ---
for /f "tokens=2 delims= " %%i in ('python --version') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if not "%PYTHON_MAJOR%.%PYTHON_MINOR%"=="3.12" (
    echo Python 3.12 ist erforderlich, aber gefunden wurde: %PYTHON_VERSION%
    pause
    exit /b 1
)

REM --- Create virtual env ---
python -m venv .venv

echo ­⚙️  Activating environment and upgrading pip...
call .venv\Scripts\activate

python -m pip install --upgrade pip

if exist requirements.txt (
    echo 📦 Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo ⚠️  No requirements.txt found. Skipping dependency install.
)

echo ­⚙️  Initializing git submodules...
git submodule update --init --recursive

echo ­📁  Creating folders...
mkdir data

echo.
echo ✅ Setup complete.
echo To activate your environment, run:
echo     .venv\Scripts\activate
pause

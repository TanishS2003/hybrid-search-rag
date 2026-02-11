@echo off
echo 🔍 Hybrid Search RAG - Quick Start
echo ====================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9-3.12
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
    echo.
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📥 Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install mmh3 first to avoid build issues on Windows
echo 📦 Installing mmh3 (fixing Windows build issue)...
pip install mmh3 --quiet

REM Install requirements
echo 📥 Installing dependencies...
echo    (This may take a few minutes on first run)
pip install -r requirements.txt --quiet

echo.
echo ✅ Setup complete!
echo.

REM Run tests
echo 🧪 Running setup tests...
python test_setup.py
echo.

echo 🚀 Starting Streamlit app...
echo    Opening at http://localhost:8501
echo.
echo    Press Ctrl+C to stop
echo.

REM Run Streamlit
streamlit run app.py

pause

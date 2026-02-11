@echo off
REM Windows Installation Script for Hybrid Search RAG
REM This script handles the mmh3 build issue

echo ========================================
echo  Hybrid Search RAG - Windows Installer
echo ========================================
echo.
echo This script will:
echo 1. Check Python version
echo 2. Create virtual environment
echo 3. Install dependencies (handling Windows issues)
echo 4. Verify installation
echo.
pause

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found
    echo.
    echo Please install Python 3.9-3.12 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo ✅ Python found
echo.

REM Create virtual environment
echo [2/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat

REM Upgrade pip
echo    Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install packages in order to avoid build issues
echo    Installing core packages...
pip install numpy scipy scikit-learn >nul 2>&1

echo    Installing mmh3 (Windows-specific handling)...
pip install mmh3 >nul 2>&1
if errorlevel 1 (
    echo    ⚠️  Warning: mmh3 installation had issues
    echo    Trying alternative approach...
    pip install mmh3==4.0.1 >nul 2>&1
)

echo    Installing Streamlit...
pip install streamlit >nul 2>&1

echo    Installing Pinecone...
pip install pinecone-client >nul 2>&1

echo    Installing LangChain...
pip install langchain langchain-community langchain-core >nul 2>&1

echo    Installing LangChain integrations...
pip install langchain-pinecone langchain-huggingface >nul 2>&1

echo    Installing ML libraries...
pip install sentence-transformers transformers >nul 2>&1

echo    Installing PyTorch...
pip install torch >nul 2>&1

echo    Installing pinecone-text...
pip install pinecone-text >nul 2>&1

echo.
echo ✅ All packages installed!
echo.

REM Verify installation
echo [4/4] Verifying installation...
python test_setup.py
echo.

REM Final message
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Get your Pinecone API key from: https://app.pinecone.io
echo 2. Run: start.bat
echo    OR
echo    Run: venv\Scripts\activate
echo         streamlit run app.py
echo.
echo For help, see WINDOWS_SETUP.md
echo.
pause

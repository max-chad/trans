@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\pythonw.exe"

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
    if not exist "%PYTHON_EXE%" (
        echo Virtual environment not found: %VENV_DIR%
        echo Create it with: python -m venv .venv
        echo Then install dependencies: %VENV_DIR%\Scripts\python.exe -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

pushd "%PROJECT_ROOT%"
"%PYTHON_EXE%" "%PROJECT_ROOT%main.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Application exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%


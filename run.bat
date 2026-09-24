@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=psdenv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

rem 1) Create the virtual environment (PyTorch needs Python 3.9 - 3.12).
if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    set "BASE_PY="
    for %%V in (3.11 3.12 3.10 3.9) do (
        if not defined BASE_PY (
            py -%%V -c "import sys" >nul 2>&1 && set "BASE_PY=py -%%V"
        )
    )
    if not defined BASE_PY (
        python -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "BASE_PY=python"
    )
    if not defined BASE_PY (
        echo Python 3.9 - 3.12 is required. Install it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    call %%BASE_PY%% -m venv "%VENV_DIR%" || goto :error
)

rem 2) Install requirements when requirements.txt changed since the last install.
fc /b requirements.txt "%VENV_DIR%\.requirements" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    "%VENV_PY%" -m pip install --upgrade pip || goto :error
    "%VENV_PY%" -m pip install -r requirements.txt || goto :error
    copy /y requirements.txt "%VENV_DIR%\.requirements" >nul
)

rem 3) Download any missing models.
"%VENV_PY%" main.py --download-models || goto :error

rem 4) Run. With no arguments this opens the web UI.
"%VENV_PY%" main.py %*
set "EXIT_CODE=%ERRORLEVEL%"
goto :end

:error
echo.
echo Setup failed, see the messages above.
pause
exit /b 1

:end
pause
exit /b %EXIT_CODE%

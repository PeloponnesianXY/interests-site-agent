@echo off
setlocal

cd /d "%~dp0"
set "DEBUG=false"
set "HOST=127.0.0.1"
set "PORT=8000"
set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if not defined PYTHON_EXE (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
  )
)

if not defined PYTHON_EXE (
  where python >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=python"
  )
)

if not defined PYTHON_EXE (
  echo Python was not found.
  echo Install Python or create a local .venv for this project.
  echo.
  pause
  exit /b 1
)

call :find_port
if errorlevel 1 (
  echo No free port found in the 8000-8010 range.
  echo Close an existing Chainlit window or free one of those ports.
  echo.
  pause
  exit /b 1
)

echo Starting Chainlit on http://%HOST%:%PORT%
echo Using Python launcher: %PYTHON_EXE%

echo Press Ctrl+C in this window to stop it.
echo.

"%PYTHON_EXE%" -m chainlit run chainlit_app.py --host %HOST% --port %PORT% --headless
if errorlevel 1 (
  echo.
  echo Chainlit failed to start.
  pause
)

endlocal
exit /b 0

:find_port
for %%P in (8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010) do (
  netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul
  if errorlevel 1 (
    set "PORT=%%P"
    exit /b 0
  )
)
exit /b 1

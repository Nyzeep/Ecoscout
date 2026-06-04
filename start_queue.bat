@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "VENV_DIR="
set "VENV_PYTHON="
set "APP_PORT=8010"

echo [1/4] Checking Python virtual environment...

if not defined VENV_DIR (
    for %%D in (.venv .venv311 venv) do (
        if not defined VENV_DIR (
            if exist "%%~D\Scripts\python.exe" if exist "%%~D\pyvenv.cfg" (
                set "VENV_DIR=%%~D"
            )
        )
    )
)

if not defined VENV_DIR (
    for /d %%D in (*) do (
        if not defined VENV_DIR (
            if exist "%%~fD\Scripts\python.exe" if exist "%%~fD\pyvenv.cfg" (
                set "VENV_DIR=%%~D"
            )
        )
    )
)

if not defined VENV_DIR (
    echo [ERROR] No project virtual environment found.
    echo Please download or copy the complete project directory, including .venv311/.venv/venv.
    echo This offline startup script will not create a virtual environment or download dependencies on site.
    pause
    exit /b 1
) else (
    echo Using virtual environment: !VENV_DIR!
)

for %%I in ("!VENV_DIR!\Scripts\python.exe") do set "VENV_PYTHON=%%~fI"

"!VENV_PYTHON!" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The project virtual environment is incomplete or dependencies are missing.
    echo Please use the prepared offline package with dependencies already installed.
    echo This script intentionally skips pip install to avoid on-site network downloads.
    pause
    exit /b 1
)

echo [2/4] Cleaning stale backend processes...
powershell -NoProfile -Command "$procs=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and (($_.CommandLine -like '*app.main:app*') -or ($_.CommandLine -like '*app.celery_app*')) }; foreach($proc in $procs){ try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop; Write-Output ('Stopped PID ' + $proc.ProcessId) } catch {} }"
timeout /t 1 /nobreak >nul

echo [3/4] Starting Celery Worker in background...
for /f %%P in ('
    powershell -NoProfile -Command "$p=Start-Process -WindowStyle Hidden -FilePath '!VENV_PYTHON!' -ArgumentList @('-m','celery','-A','app.celery_app','worker','--loglevel=info','--pool=solo') -PassThru; $p.Id"
') do set "CELERY_PID=%%P"

timeout /t 2 /nobreak >nul

echo [4/4] Starting FastAPI Web in background...
for /f %%P in ('
    powershell -NoProfile -Command "$p=Start-Process -WindowStyle Hidden -FilePath '!VENV_PYTHON!' -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','!APP_PORT!') -PassThru; $p.Id"
') do set "UVICORN_PID=%%P"

timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:!APP_PORT!

echo.
echo Started successfully:
echo - Virtual environment: !VENV_DIR!
echo - Celery PID: !CELERY_PID!
echo - Uvicorn PID: !UVICORN_PID!
echo - Browser: http://127.0.0.1:!APP_PORT!
echo.
echo Press any key to stop all services...
pause >nul

echo Stopping services...
if defined UVICORN_PID taskkill /PID !UVICORN_PID! /T /F >nul 2>&1
if defined CELERY_PID taskkill /PID !CELERY_PID! /T /F >nul 2>&1
echo All services stopped.

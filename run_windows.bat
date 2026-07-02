@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

set "VIDEO_PATH=%~1"
if "%VIDEO_PATH%"=="" set /p "VIDEO_PATH=請輸入影片完整路徑: "

.venv\Scripts\python.exe main.py --video "%VIDEO_PATH%"
endlocal

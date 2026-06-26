@echo off
REM Launch the demo dashboard using the MAIN project's virtualenv (no separate install).
cd /d "%~dp0"
"C:\AEROS UPC\4A\PD\proyecto-drones-tello\.venv\Scripts\python.exe" demo_dashboard.py %*
pause

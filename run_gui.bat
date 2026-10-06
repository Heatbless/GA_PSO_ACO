@echo off
cd /d "%~dp0"
python -m sppg_compare
if errorlevel 1 pause

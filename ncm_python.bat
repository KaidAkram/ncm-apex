@echo off
rem NCM-Apex Environment Runner with workspace root in PYTHONPATH and UTF-8 encoding
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
"C:\anaconda3\envs\gaussian_splatting\python.exe" %*

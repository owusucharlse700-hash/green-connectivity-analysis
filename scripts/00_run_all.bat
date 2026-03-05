@echo off
setlocal

cd /d "%~dp0\.."
echo [INFO] Running from repo root: %CD%

python scripts\01_graph_metrics.py
if errorlevel 1 goto :error
echo [OUT] Graph outputs saved under outputs\graph\

Rscript scripts\02_makurhini_pc_iic_eca.R
if errorlevel 1 goto :error
echo [OUT] Makurhini outputs saved under outputs\makurhini\

python scripts\03_circuit_metrics.py
if errorlevel 1 goto :error
echo [OUT] Circuit outputs saved under outputs\circuit\

echo [DONE] All workflows completed.
goto :eof

:error
echo [ERROR] Pipeline stopped due to failure.
exit /b 1

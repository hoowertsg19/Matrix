@echo off
setlocal
set ROOT=%~dp0
set PY=%ROOT%\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Creando entorno virtual...
  py -m venv "%ROOT%\.venv"
)
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
"%PY%" -m pip install --upgrade pyinstaller pillow

REM Generar logo.ico a partir de logo.png si no existe
if not exist "%ROOT%\logo.ico" (
  echo Creando icono logo.ico...
  "%PY%" "%ROOT%\make_icon.py"
)

"%PY%" -m PyInstaller "%ROOT%\MatrixQt.spec"
echo.
echo Listo. Ejecutable en: %ROOT%\dist\Matrix.exe
endlocal

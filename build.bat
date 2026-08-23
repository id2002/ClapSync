@echo off
REM ============================================================
REM build.bat
REM Empaqueta clap_sync_gui.py en un solo ClapSync.exe (Windows).
REM Correr este archivo DESDE la misma carpeta donde esta
REM clap_sync_gui.py, haciendo doble clic o desde una terminal.
REM ============================================================

echo Instalando dependencias (solo la primera vez tarda)...
python -m pip install --upgrade pip
python -m pip install pyinstaller numpy opencv-python scipy

if %errorlevel% neq 0 (
    echo.
    echo Algo fallo instalando dependencias. Revisa que Python este
    echo instalado y agregado al PATH ^(prueba: python --version^).
    pause
    exit /b 1
)

echo.
echo Generando ClapSync.exe... esto puede tardar unos minutos.
pyinstaller --onefile --noconsole --name ClapSync clap_sync_gui.py

echo.
echo ============================================================
echo Listo! Busca ClapSync.exe dentro de la carpeta "dist"
echo que se creo junto a este archivo.
echo ============================================================
pause

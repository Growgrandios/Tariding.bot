@echo off
REM Trading Bot Startup-Skript für Windows

REM Konfiguration
set BOT_DIR=%~dp0
set VENV_DIR=%BOT_DIR%\venv
set LOG_DIR=%BOT_DIR%\logs
set CONFIG_FILE=%BOT_DIR%\data\config\config.yaml

REM Verzeichnis wechseln
cd /d "%BOT_DIR%" || (
    echo FEHLER: Konnte nicht ins Bot-Verzeichnis wechseln
    exit /b 1
)

REM Prüfen, ob virtuelle Umgebung existiert
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtuelle Umgebung nicht gefunden. Erstelle neue Umgebung...
    python -m venv "%VENV_DIR%" || (
        echo FEHLER: Konnte virtuelle Umgebung nicht erstellen
        exit /b 1
    )
)

REM Virtuelle Umgebung aktivieren
call "%VENV_DIR%\Scripts\activate.bat" || (
    echo FEHLER: Konnte virtuelle Umgebung nicht aktivieren
    exit /b 1
)

REM Abhängigkeiten installieren/aktualisieren
pip install -r requirements.txt

REM Verzeichnisse erstellen
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "data\config" mkdir "data\config"

REM Trading Bot starten
echo Starte Trading Bot...
python main.py --config "%CONFIG_FILE%" %*

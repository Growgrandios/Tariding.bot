#!/bin/bash
# Trading Bot Startup-Skript für Linux/Mac

# Konfiguration
BOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${BOT_DIR}/venv"
LOG_DIR="${BOT_DIR}/logs"
CONFIG_FILE="${BOT_DIR}/data/config/config.yaml"

# Funktion zur Fehlerbehandlung
handle_error() {
    echo "FEHLER: $1"
    exit 1
}

# Verzeichnis wechseln
cd "${BOT_DIR}" || handle_error "Konnte nicht ins Bot-Verzeichnis wechseln"

# Prüfen, ob virtuelle Umgebung existiert
if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtuelle Umgebung nicht gefunden. Erstelle neue Umgebung..."
    python3 -m venv "${VENV_DIR}" || handle_error "Konnte virtuelle Umgebung nicht erstellen"
fi

# Virtuelle Umgebung aktivieren
source "${VENV_DIR}/bin/activate" || handle_error "Konnte virtuelle Umgebung nicht aktivieren"

# Abhängigkeiten installieren/aktualisieren
pip install -r requirements.txt

# Verzeichnisse erstellen
mkdir -p "${LOG_DIR}"
mkdir -p data/config

# Trading Bot starten
echo "Starte Trading Bot..."
python main.py --config "${CONFIG_FILE}" "$@"

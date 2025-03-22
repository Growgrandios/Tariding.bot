#!/bin/bash

# Trading Bot Startup-Skript für Linux/Mac mit NVIDIA GPU-Unterstützung

# Konfiguration
BOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${BOT_DIR}/venv"
LOG_DIR="${BOT_DIR}/logs"
CONFIG_FILE="${BOT_DIR}/data/config/config.yaml"
CONFIG_SAMPLE="${BOT_DIR}/data/config/config_sample.yaml"

# Funktion zur Fehlerbehandlung
handle_error() {
    echo "FEHLER: $1"
    exit 1
}

# Überprüfe NVIDIA-GPU und CUDA
check_nvidia() {
    echo "Prüfe NVIDIA GPU-Unterstützung..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU erkannt:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -n 1)
        echo "CUDA Version: $CUDA_VERSION"
        return 0
    else
        echo "WARNUNG: NVIDIA GPU nicht erkannt oder nvidia-smi nicht installiert."
        echo "         Nur CPU-Betrieb möglich."
        return 1
    fi
}

# System-Abhängigkeiten prüfen und installieren
check_system_dependencies() {
    echo "Prüfe System-Abhängigkeiten..."
    
    # Prüfe ob ta-lib installiert ist
    if ! ldconfig -p | grep -q "libta_lib"; then
        echo "TA-Lib nicht installiert. Versuche zu installieren..."
        sudo apt-get update
        sudo apt-get install -y build-essential wget
        
        # TA-Lib herunterladen und installieren
        cd /tmp
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        cd "${BOT_DIR}"
    fi
}

# Verzeichnis wechseln
cd "${BOT_DIR}" || handle_error "Konnte nicht ins Bot-Verzeichnis wechseln"

# Prüfe NVIDIA GPU
check_nvidia

# Systemabhängigkeiten prüfen
check_system_dependencies

# Erstelle alle notwendigen Verzeichnisse
echo "Erstelle Verzeichnisstruktur..."
mkdir -p "${LOG_DIR}"
mkdir -p data/config
mkdir -p data/models
mkdir -p data/backtest_results
mkdir -p data/knowledge
mkdir -p data/transcripts
mkdir -p data/tax/reports
mkdir -p data/black_swan

# Prüfen, ob virtuelle Umgebung existiert
if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtuelle Umgebung nicht gefunden. Erstelle neue Umgebung..."
    python3 -m venv "${VENV_DIR}" || handle_error "Konnte virtuelle Umgebung nicht erstellen"
fi

# Virtuelle Umgebung aktivieren
source "${VENV_DIR}/bin/activate" || handle_error "Konnte virtuelle Umgebung nicht aktivieren"

# Abhängigkeiten installieren/aktualisieren
echo "Installiere/aktualisiere Python-Abhängigkeiten..."
pip install --upgrade pip || handle_error "Konnte pip nicht aktualisieren"
pip install -r requirements.txt || handle_error "Konnte Abhängigkeiten nicht installieren"

# Prüfen, ob Konfigurationsdatei existiert
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Konfigurationsdatei nicht gefunden."
    if [ -f "${CONFIG_SAMPLE}" ]; then
        echo "Kopiere Beispielkonfiguration..."
        cp "${CONFIG_SAMPLE}" "${CONFIG_FILE}" || handle_error "Konnte Beispielkonfiguration nicht kopieren"
    else
        echo "Keine Beispielkonfiguration gefunden. Bot wird eine Standardkonfiguration erstellen."
    fi
fi

# Umgebungsvariablen prüfen
if [ ! -f "${BOT_DIR}/.env" ]; then
    echo "Keine .env-Datei gefunden. Erstelle leere Datei..."
    touch "${BOT_DIR}/.env"
    echo "# API-Schlüssel" >> "${BOT_DIR}/.env"
    echo "BITGET_API_KEY=" >> "${BOT_DIR}/.env"
    echo "BITGET_API_SECRET=" >> "${BOT_DIR}/.env"
    echo "BITGET_API_PASSPHRASE=" >> "${BOT_DIR}/.env"
    echo "TELEGRAM_BOT_TOKEN=" >> "${BOT_DIR}/.env"
    echo "ALPHA_VANTAGE_API_KEY=" >> "${BOT_DIR}/.env"
    echo "NEWS_API_KEY=" >> "${BOT_DIR}/.env"
    echo "HUGGINGFACE_TOKEN=" >> "${BOT_DIR}/.env"
    echo "Bitte API-Schlüssel in .env-Datei eintragen."
fi

# Trading Bot starten
echo "Starte Trading Bot..."
python main.py --config "${CONFIG_FILE}" "$@"

# main.py

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Verzeichnisse zuerst einrichten
def setup_directories():
    """Erstellt die notwendigen Verzeichnisse, falls sie nicht existieren."""
    directories = [
        "logs",
        "data",
        "data/config",
        "data/models",
        "data/backtest_results",
        "data/knowledge",
        "data/transcripts",
        "data/tax",
        "data/tax/reports",
        "data/black_swan"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Logs-Verzeichnis vor dem Einrichten des Loggings erstellen
setup_directories()

# Logging-System konfigurieren
from src.utils.logging_setup import setup_logging
logger = setup_logging(log_level=logging.INFO)

# Core-Module importieren (nach Logging-Setup)
from src.core.config_manager import ConfigManager
from src.core.main_controller import MainController

def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description="Gemma Trading Bot - Ein auf Gemma 3 basierender Krypto-Trading-Bot")
    parser.add_argument("--config", type=str, default="data/config/config.yaml",
                        help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--mode", type=str, choices=["live", "paper", "backtest", "learn"],
                        help="Trading-Modus (überschreibt Konfigurationsdatei)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug-Modus aktivieren")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Telegram-Bot deaktivieren")
    parser.add_argument("--process-transcript", type=str,
                        help="Transkriptdatei verarbeiten und beenden")
    parser.add_argument("--learn", action="store_true",
                        help="Starten im Lernmodus (trainiere Modelle)")
    return parser.parse_args()

def main():
    """Hauptfunktion des Trading Bots."""
    # Start-Nachricht
    logger.info("=======================================")
    logger.info("Gemma Trading Bot wird gestartet...")
    logger.info("=======================================")
    
    # Lade Umgebungsvariablen aus .env-Datei
    load_dotenv()
    
    # Kommandozeilenargumente parsen
    args = parse_arguments()
    
    # Debug-Modus
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug-Modus aktiviert")
    
    # Config Manager initialisieren
    config_manager = ConfigManager(args.config)
    
    # Main Controller initialisieren
    controller = MainController(config_manager)
    
    # Prüfen, ob Transkript-Verarbeitung angefordert wurde
    if args.process_transcript:
        logger.info(f"Verarbeite Transkript: {args.process_transcript}")
        result = controller.process_transcript(args.process_transcript)
        logger.info(f"Ergebnis: {json.dumps(result, indent=2)}")
        return
    
    # Prüfen, ob der Lernmodus aktiviert ist
    if args.learn:
        logger.info("Starte im Lernmodus")
        controller.start(mode="learn", auto_trade=False)
        controller.train_models()
        controller.stop()
        logger.info("Lernen abgeschlossen, Programm wird beendet")
        return
    
    # Trading-Modus überschreiben, falls angegeben
    mode = args.mode if args.mode else config_manager.get_config("trading").get("mode", "paper")
    logger.info(f"Trading-Modus: {mode}")
    
    # Telegram deaktivieren, falls angefordert
    if args.no_telegram:
        logger.info("Telegram-Bot wird deaktiviert")
        config = config_manager.get_config()
        if "telegram" in config:
            config["telegram"]["enabled"] = False
            config_manager.update_section("telegram", config["telegram"])
    
    try:
        # Bot mit entsprechendem Modus starten
        controller.start(mode=mode, auto_trade=(mode == "live"))
        
        # Hauptthread am Leben halten
        # Annahme: controller.start() ist nicht-blockierend und wir müssen das Programm am Laufen halten
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Benutzerabbruch erkannt")
    except Exception as e:
        logger.error(f"Fehler im Hauptprogramm: {str(e)}", exc_info=True)
    finally:
        # Alles sauber beenden
        controller.stop()
        logger.info("Gemma Trading Bot beendet")

if __name__ == "__main__":
    main()

# main.py

import os
import sys
import logging
import traceback
import json
from datetime import datetime
from pathlib import Path

# Import des Telegram-Moduls
from telegram_module import TelegramBot

# Konfiguration des Loggings
def setup_logging(level=logging.INFO):
    """Richtet das Logging-System ein"""
    # Stelle sicher, dass das Logs-Verzeichnis existiert
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log-Dateiname mit Datum
    log_filename = log_dir / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Konfiguriere das Root-Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Formatter für detaillierte Ausgabe
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler für Konsole
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Handler für Datei
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

def load_config(config_file='config.json'):
    """Lädt die Konfigurationsdatei"""
    logger = logging.getLogger("Config")
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Konfiguration aus {config_file} geladen")
                return config
        else:
            logger.warning(f"Konfigurationsdatei {config_file} nicht gefunden. Verwende Standardwerte.")
            # Standard-Konfiguration
            return {
                "telegram_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                "admin_ids": [],
                "log_level": "INFO"
            }
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def main():
    """Hauptfunktion zum Starten des Bots"""
    # Logging einrichten
    logger = setup_logging()
    logger.info("=== BOT STARTET ===")
    
    try:
        # Konfiguration laden
        config = load_config()
        
        # Telegram-Token überprüfen
        telegram_token = config.get('telegram_token', os.environ.get("TELEGRAM_BOT_TOKEN"))
        if not telegram_token:
            logger.error("Kein Telegram-Token gefunden. Bitte in config.json oder als Umgebungsvariable TELEGRAM_BOT_TOKEN angeben.")
            sys.exit(1)
        
        # Admin-IDs (können für spezielle Benachrichtigungen verwendet werden)
        admin_ids = config.get('admin_ids', [])
        
        # Log-Level aus Konfiguration
        log_level_str = config.get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        # Stelle Root-Logger auf das richtige Level ein
        logging.getLogger().setLevel(log_level)
        
        logger.info(f"Konfiguration geladen: Log-Level={log_level_str}, Admins={admin_ids}")
        
        # Telegram-Bot initialisieren
        logger.info("Initialisiere Telegram-Bot...")
        bot = TelegramBot(
            token=telegram_token,
            admin_ids=admin_ids,
            log_level=log_level
        )
        
        # Registriere benutzerdefinierte Handler (Beispiel)
        # async def handle_stats(update, context):
        #     await update.message.reply_text("Hier sind deine Statistiken...")
        # bot.register_command_handler("stats", handle_stats)
        
        # Bot starten
        logger.info("Starte Bot...")
        webhook_url = config.get('webhook_url')  # Optional für Webhook-Modus
        bot.run(webhook_url)
        
    except KeyboardInterrupt:
        logger.info("Bot durch Benutzer gestoppt (KeyboardInterrupt)")
    except Exception as e:
        logger.critical(f"Unerwarteter Fehler: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

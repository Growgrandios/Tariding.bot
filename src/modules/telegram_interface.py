# telegram_interface.py
import os
import logging
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable

# Importiere Telegram-Module mit der alten API-Struktur
from telegram import __version__ as TG_VER
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    ParseMode  # Direkter Import in Version 13.x
)
from telegram.ext import (
    Updater,  # Die alte API verwendet Updater statt Application
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    Filters,  # Großbuchstaben in der alten API
    CallbackContext,
    Dispatcher,
    ConversationHandler
)

# Google Cloud VM Management
from google.cloud import compute_v1

# Eigene Module integrieren 
# Importpfade angepasst, um verschiedene Strukturen zu unterstützen
try:
    from src.modules.data_pipeline import DataPipeline
except ImportError:
    try:
        from data_pipeline import DataPipeline
    except ImportError:
        DataPipeline = None
        logging.warning("DataPipeline-Modul nicht gefunden")

try:
    from src.modules.transcript_processor import TranscriptProcessor
except ImportError:
    try:
        from transcript_processor import TranscriptProcessor
    except ImportError:
        TranscriptProcessor = None
        logging.warning("TranscriptProcessor-Modul nicht gefunden")

try:
    from src.modules.live_trading import LiveTradingConnector
except ImportError:
    try:
        from live_trading import LiveTradingConnector
    except ImportError:
        LiveTradingConnector = None
        logging.warning("LiveTradingConnector-Modul nicht gefunden")

try:
    from src.modules.black_swan_detector import BlackSwanDetector
except ImportError:
    try:
        from black_swan_detector import BlackSwanDetector
    except ImportError:
        BlackSwanDetector = None
        logging.warning("BlackSwanDetector-Modul nicht gefunden")

# Logging Konfiguration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TelegramInterface")

class TelegramInterface:
    """Master-Interface für Trading-Bot Steuerung via Telegram"""
    
    # Konversationszustände
    STRATEGY_SELECT, RISK_MANAGEMENT, ORDER_CONFIRM = range(3)
    
    def __init__(self, config: Dict, trading_modules: Dict = None):
        """
        Initialisiert das Telegram Interface.
        
        Args:
            config: Dictionary mit Konfigurationseinstellungen
            trading_modules: Optional, Dictionary mit Trading-Modulen
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface...")
        
        # Konfiguration speichern
        self.config = config or {}
        self.modules = trading_modules or {}
        
        # API Token prüfen
        self.token = self.config.get('token', os.getenv('TELEGRAM_TOKEN'))
        if not self.token:
            self.logger.error("Telegram API-Token fehlt. Setze TELEGRAM_TOKEN in der Konfiguration oder .env-Datei.")
            raise ValueError("Telegram API-Token fehlt")
        
        # Initialisiere Telegram Bot
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # Google Cloud Client
        gcp_project = self.config.get('gcp_project', os.getenv('GCP_PROJECT_ID'))
        gcp_zone = self.config.get('gcp_zone', os.getenv('GCP_ZONE', 'europe-west3-a')) # Google Cloud Region für Deutschland
        gcp_instance = self.config.get('gcp_instance', os.getenv('GCP_INSTANCE', 'trading-bot-vm'))
        
        if gcp_project:
            try:
                self.vm_client = compute_v1.InstancesClient()
                self.project_id = gcp_project
                self.zone = gcp_zone
                self.instance_name = gcp_instance
                self.gcp_enabled = True
                self.logger.info(f"Google Cloud VM Management aktiviert für Projekt {self.project_id}")
            except Exception as e:
                self.logger.error(f"Fehler bei der Initialisierung des GCP-Clients: {str(e)}")
                self.gcp_enabled = False
        else:
            self.gcp_enabled = False
            self.logger.warning("Google Cloud VM Management deaktiviert (fehlende Konfiguration)")
        
        # Sicherheitskonfiguration
        self.allowed_users = set(self.config.get('allowed_users', []))
        self.admin_users = set(self.config.get('admin_users', []))
        
        # Status-Tracking
        self.bot_active = False
        self.last_activity = datetime.now()
        
        # Register handlers
        self._register_handlers()
        
        self.logger.info("TelegramInterface erfolgreich initialisiert")
    
    def _register_handlers(self):
        """Registriert alle Telegram-Handler"""
        # Befehlshandler
        self.dispatcher.add_handler(CommandHandler("start", self.cmd_start))
        self.dispatcher.add_handler(CommandHandler("help", self.cmd_help))
        self.dispatcher.add_handler(CommandHandler("status", self.cmd_status))
        
        # VM-Management
        if self.gcp_enabled:
            self.dispatcher.add_handler(CommandHandler("vm_start", self.cmd_vm_start))
            self.dispatcher.add_handler(CommandHandler("vm_stop", self.cmd_vm_stop))
            self.dispatcher.add_handler(CommandHandler("vm_status", self.cmd_vm_status))
        
        # Notfall-Kommandos
        self.dispatcher.add_handler(CommandHandler("emergency_stop", self.cmd_emergency_stop))
        
        # Callback-Handler für Inline-Keyboard
        self.dispatcher.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Fehlerhandler
        self.dispatcher.add_error_handler(self.error_handler)
        
        self.logger.info("Telegram-Handler registriert")

    # Hier weitere Methoden wie cmd_start, cmd_vm_start, etc.
    # Diese Implementierungen würden zu weit führen für diese Antwort
    
    def start(self):
        """Startet den Telegram Bot"""
        self.logger.info("Starte TelegramInterface...")
        self.updater.start_polling()
        self.last_activity = datetime.now()
        self.logger.info("TelegramInterface erfolgreich gestartet")
    
    def stop(self):
        """Stoppt den Telegram Bot"""
        self.logger.info("Stoppe TelegramInterface...")
        self.updater.stop()
        self.logger.info("TelegramInterface erfolgreich gestoppt")

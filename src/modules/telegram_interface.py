 telegram_interface.py
import os
import logging
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable

from telegram import __version__ as TG_VER
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
# Neuer Import für ParseMode
from telegram.constants import ParseMode
from telegram.ext import (
    Application,  # Statt Updater
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,  # Kleinbuchstabe statt Filters
    ContextTypes,  # Statt CallbackContext
    ConversationHandler
)

# Google Cloud VM Management
from google.cloud import compute_v1

# Eigene Module integrieren
from data_pipeline import DataPipeline
from transcript_processor import TranscriptProcessor
from live_trading import LiveTradingConnector
from black_swan_detector import BlackSwanDetector

# Logging Konfiguration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TelegramInterface")

class TelegramTradingInterface:
    """Master-Interface für Trading-Bot Steuerung via Telegram"""
    
    # Konversationszustände
    STRATEGY_SELECT, RISK_MANAGEMENT, ORDER_CONFIRM = range(3)
    
    def __init__(self, config: Dict, trading_modules: Dict):
        self.config = config
        self.modules = trading_modules
        
        # Initialisiere Telegram Bot
        self.updater = Updater(token=config['TELEGRAM_TOKEN'], use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # Google Cloud Client
        self.vm_client = compute_v1.InstancesClient()
        self.project_id = config['GCP_PROJECT_ID']
        self.zone = config['GCP_ZONE']
        self.instance_name = config['GCP_INSTANCE_NAME']
        
        # Sicherheitskonfiguration
        self.allowed_users = set(config['ALLOWED_USER_IDS'])
        self.admin_users = set(config['ADMIN_USER_IDS'])
        
        # Registriere Handler
        self._register_handlers()
        self._set_bot_commands()
        
        # Status Tracking
        self.user_sessions = {}
        self.last_alerts = {}
        self.notification_queue = []

        # Starte Nachrichten-Queue Worker
        self.queue_worker = threading.Thread(target=self._process_notification_queue)
        self.queue_worker.daemon = True
        self.queue_worker.start()

    def _register_handlers(self):
        """Registriere alle Telegram Handler"""
        handlers = [
            CommandHandler('start', self.start),
            CommandHandler('emergency_stop', self.emergency_stop),
            CommandHandler('vm_control', self.vm_control),
            CommandHandler('report', self.generate_report),
            CallbackQueryHandler(self.button_handler),
            MessageHandler(Filters.text & ~Filters.command, self.message_handler)
        ]
        
        # Konversationshandler für Trading
        trade_conv = ConversationHandler(
            entry_points=[CommandHandler('trade', self.start_trade)],
            states={
                self.STRATEGY_SELECT: [CallbackQueryHandler(self.strategy_select)],
                self.RISK_MANAGEMENT: [MessageHandler(Filters.text, self.risk_management)],
                self.ORDER_CONFIRM: [CallbackQueryHandler(self.order_confirm)]
            },
            fallbacks=[CommandHandler('cancel', self.cancel_trade)]
        )
        handlers.append(trade_conv)

        for handler in handlers:
            self.dispatcher.add_handler(handler)

    def _set_bot_commands(self):
        """Setze Bot-Befehle für UI"""
        commands = [
            BotCommand('start', 'Starte den Trading Bot'),
            BotCommand('trade', 'Starte neuen Trade'),
            BotCommand('report', 'Erhalte aktuellen Report'),
            BotCommand('vm_control', 'VM Instanz steuern'),
            BotCommand('emergency_stop', 'Notfall-Stopp aller Aktivitäten')
        ]
        self.updater.bot.set_my_commands(commands)

    def _authorized(self, func: Callable) -> Callable:
        """Decorator für autorisierten Zugriff"""
        def wrapper(update: Update, context: CallbackContext):
            user_id = update.effective_user.id
            if user_id not in self.allowed_users:
                update.message.reply_text("⛔ Zugriff verweigert. Nicht autorisierter Benutzer.")
                logger.warning(f"Unautorisierter Zugriff von User-ID: {user_id}")
                return
            return func(update, context)
        return wrapper

    # --- Kernfunktionen ---
    @_authorized
    def start(self, update: Update, context: CallbackContext):
        """Startkommando mit interaktivem Dashboard"""
        keyboard = [
            [InlineKeyboardButton("🚀 Live Trading", callback_data='live_dashboard'),
             InlineKeyboardButton("📈 Strategien", callback_data='strategies')],
            [InlineKeyboardButton("⚠️ Notfall-Stopp", callback_data='emergency'),
             InlineKeyboardButton("📊 Reports", callback_data='reports')],
            [InlineKeyboardButton("⚙️ Einstellungen", callback_data='settings')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(
            "🤖 *Trading Bot Kontrollzentrum* 🤖\n"
            "Wähle eine Option:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )

    @_authorized
    def emergency_stop(self, update: Update, context: CallbackContext):
        """Notfall-Stopp aller Trading-Aktivitäten"""
        self.modules['live_trading'].stop_trading()
        self.modules['data_pipeline'].stop_auto_updates()
        
        # Alle offenen Positionen schließen
        self.close_all_positions()
        
        update.message.reply_text(
            "🛑 *NOTFALL-STOPP AKTIVIERT* 🛑\n"
            "Alle Trading-Aktivitäten wurden gestoppt!",
            parse_mode=ParseMode.MARKDOWN
        )

    def _process_notification_queue(self):
        """Verarbeite die Nachrichten-Queue für Echtzeit-Updates"""
        while True:
            while self.notification_queue:
                msg, priority = self.notification_queue.pop(0)
                try:
                    for user in self.admin_users:
                        self.updater.bot.send_message(
                            chat_id=user,
                            text=msg,
                            parse_mode=ParseMode.MARKDOWN
                        )
                except Exception as e:
                    logger.error(f"Fehler beim Senden der Benachrichtigung: {str(e)}")
            time.sleep(1)

    # --- VM Management ---
    @_authorized
    def vm_control(self, update: Update, context: CallbackContext):
        """Steuere die Google Cloud VM Instanz"""
        keyboard = [
            [InlineKeyboardButton("▶️ VM Starten", callback_data='vm_start'),
             InlineKeyboardButton("⏹️ VM Stoppen", callback_data='vm_stop')],
            [InlineKeyboardButton("🔄 Status Abfragen", callback_data='vm_status')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(
            "☁️ *Google Cloud VM Management* ☁️\n"
            "Wähle eine Aktion:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )

    def _vm_operation(self, operation: str):
        """Führe VM-Operationen aus"""
        ops = {
            'start': self.vm_client.start,
            'stop': self.vm_client.stop,
            'status': self.vm_client.get
        }
        try:
            result = ops[operation](
                project=self.project_id,
                zone=self.zone,
                instance=self.instance_name
            )
            return f"VM {operation} erfolgreich: {result.status}"
        except Exception as e:
            logger.error(f"VM Operation fehlgeschlagen: {str(e)}")
            return f"❌ Fehler bei VM {operation}: {str(e)}"

    # --- Trading-Funktionen ---
    def start_trade(self, update: Update, context: CallbackContext):
        """Starte Trading-Konversation"""
        strategies = self.modules['strategy_manager'].get_available_strategies()
        keyboard = [
            [InlineKeyboardButton(s['name'], callback_data=s['id'])]
            for s in strategies
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(
            "📊 Verfügbare Strategien:\n"
            "Wähle eine Trading-Strategie:",
            reply_markup=reply_markup
        )
        return self.STRATEGY_SELECT

    def strategy_select(self, update: Update, context: CallbackContext):
        """Verarbeite Strategieauswahl"""
        query = update.callback_query
        strategy_id = query.data
        context.user_data['strategy_id'] = strategy_id
        
        query.edit_message_text(
            text=f"Gewählte Strategie: {strategy_id}\n"
                 "Bitte Risikoparameter eingeben (Format: Risiko% StopLoss% TakeProfit%):\n"
                 "Beispiel: 2 5 10"
        )
        return self.RISK_MANAGEMENT

    # --- Benachrichtigungssystem ---
    def send_real_time_alert(self, alert_data: Dict):
        """Sende Echtzeit-Alert an registrierte Benutzer"""
        alert_msg = (
            f"🚨 *{alert_data['type'].upper()} ALERT* 🚨\n"
            f"*Symbol*: {alert_data['symbol']}\n"
            f"*Preis*: {alert_data['price']}\n"
            f"*Nachricht*: {alert_data['message']}"
        )
        self.notification_queue.append((alert_msg, 'high'))

    # --- Integration mit anderen Modulen ---
    def register_module_callbacks(self):
        """Registriere Callbacks für andere Module"""
        # Live Trading
        self.modules['live_trading'].register_notification_callback(
            self.handle_trading_update
        )
        
        # Black Swan Detector
        self.modules['black_swan'].register_notification_callback(
            self.handle_black_swan_alert
        )

    def handle_trading_update(self, update_data: Dict):
        """Verarbeite Trading-Updates"""
        msg = (
            f"📊 *Trade Update* 📊\n"
            f"Order {update_data['status']}:\n"
            f"Symbol: {update_data['symbol']}\n"
            f"Typ: {update_data['type']}\n"
            f"Menge: {update_data['amount']}"
        )
        self.notification_queue.append((msg, 'medium'))

    def handle_black_swan_alert(self, alert_data: Dict):
        """Verarbeite Black Swan Events"""
        alert_msg = (
            f"⚠️⚫ *Black Swan Event* ⚫⚠️\n"
            f"Typ: {alert_data['details']['type']}\n"
            f"Schweregrad: {alert_data['severity']}/1.0\n"
            f"Empfehlung: {alert_data['recommendation']}"
        )
        self.notification_queue.append((alert_msg, 'critical'))

    # --- Hilfsfunktionen ---
    def _format_report(self, report_data: Dict) -> str:
        """Formatiere Trading-Bericht für Telegram"""
        return (
            "📈 *Tagesbericht* 📈\n"
            f"*Portfoliowert*: ${report_data['portfolio_value']:.2f}\n"
            f"*Heutige Trades*: {report_data['trades_today']}\n"
            f"*Gewinn/Verlust*: {report_data['pnl']}%\n"
            f"*Risikoexposure*: {report_data['risk_exposure']}%\n"
            "🔔 Aktuelle Marktbedingungen: "
            f"{report_data['market_condition']}"
        )

    def run(self):
        """Starte den Telegram Bot"""
        logger.info("Starting Telegram Interface...")
        self.updater.start_polling()
        self.updater.idle()

if __name__ == '__main__':
    config = {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID'),
        'GCP_ZONE': 'europe-west3-a',
        'GCP_INSTANCE_NAME': 'trading-bot-vm',
        'ALLOWED_USER_IDS': [12345678],
        'ADMIN_USER_IDS': [12345678]
    }
    
    # Modul-Instanzen (Beispiel)
    modules = {
        'live_trading': LiveTradingConnector(config),
        'data_pipeline': DataPipeline(config),
        'black_swan': BlackSwanDetector(config),
        'transcript_processor': TranscriptProcessor(config)
    }
    
    interface = TelegramTradingInterface(config, modules)
    interface.register_module_callbacks()
    interface.run()

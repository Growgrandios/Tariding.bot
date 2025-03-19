# telegram_interface.py

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, ContextTypes
from telegram.constants import ParseMode
import re
from pathlib import Path

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/telegram_interface.log"),
        logging.StreamHandler()
    ]
)

class TelegramInterface:
    """
    Telegram-Bot-Schnittstelle für die Fernsteuerung und Benachrichtigungen des Trading-Bots.
    Ermöglicht die Interaktion mit dem Bot über Telegram-Nachrichten.
    """
    
    def __init__(self, config: Dict[str, Any], main_controller=None):
        """
        Initialisiert die Telegram-Schnittstelle.
        
        Args:
            config: Konfigurationseinstellungen mit Bot-Token und erlaubten Benutzern
            main_controller: Referenz zum MainController für Zugriff auf andere Module
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface...")
        
        # API-Konfiguration
        self.bot_token = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        
        # String oder Liste von IDs in Integer-Liste konvertieren
        allowed_users_raw = config.get('allowed_users', [])
        if isinstance(allowed_users_raw, str):
            # String von kommagetrennten IDs parsen
            self.allowed_users = [int(user_id.strip()) for user_id in allowed_users_raw.split(',') if user_id.strip()]
        elif isinstance(allowed_users_raw, list):
            # Liste von verschiedenen Typen in Integer konvertieren
            self.allowed_users = [int(user_id) for user_id in allowed_users_raw if user_id]
        else:
            self.allowed_users = []
        
        # Prüfen, ob Token und Benutzer konfiguriert sind
        if not self.bot_token:
            self.logger.error("Kein Telegram-Bot-Token konfiguriert")
            self.is_configured = False
        elif not self.allowed_users:
            self.logger.warning("Keine erlaubten Telegram-Benutzer konfiguriert")
            self.is_configured = True  # Wir können trotzdem starten, aber keine Befehle annehmen
        else:
            self.is_configured = True
            self.logger.info(f"{len(self.allowed_users)} erlaubte Benutzer konfiguriert")
        
        # Benachrichtigungskonfiguration
        self.notification_level = config.get('notification_level', 'INFO')
        self.status_update_interval = config.get('status_update_interval', 3600)  # Sekunden
        self.commands_enabled = config.get('commands_enabled', True)
        
        # Begrenzer für Benachrichtigungen
        self.notification_cooldown = config.get('notification_cooldown', 60)  # Sekunden
        self.last_notification_time = {}  # Dict für Zeitstempel der letzten Benachrichtigung pro Priorität
        self.max_notifications_per_hour = {
            'low': config.get('max_low_priority_per_hour', 10),
            'normal': config.get('max_normal_priority_per_hour', 20),
            'high': config.get('max_high_priority_per_hour', 30),
            'critical': config.get('max_critical_priority_per_hour', 50)
        }
        self.notification_counts = {
            'low': 0,
            'normal': 0,
            'high': 0,
            'critical': 0
        }
        self.notification_reset_time = datetime.now() + timedelta(hours=1)
        
        # Hauptcontroller-Referenz
        self.main_controller = main_controller
        
        # Bot-Instanz und Application
        self.bot = None
        self.application = None
        
        # Thread für Bot-Updates
        self.bot_thread = None
        self.is_running = False
        
        # Befehlsreferenzen (für dynamische Befehle)
        self.command_handlers = {}
        
        # Verzeichnis für aufgezeichnete Transkripte
        self.transcript_dir = Path('data/transcripts')
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Bot, wenn konfiguriert
        if self.is_configured:
            self._setup_bot()
        
        self.logger.info("TelegramInterface erfolgreich initialisiert")
    
    def _setup_bot(self):
        """Richtet den Telegram-Bot mit Befehlen und Handlern ein."""
        try:
            # Bot-Instanz erstellen
            self.bot = Bot(token=self.bot_token)
            
            # Application erstellen
            self.application = Application.builder().token(self.bot_token).build()
            
            # Standard-Befehlshandler hinzufügen
            self._add_default_handlers()
            
            self.logger.info("Telegram-Bot erfolgreich eingerichtet")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Einrichten des Telegram-Bots: {str(e)}")
            self.is_configured = False
    
    def _add_default_handlers(self):
        """Fügt Standardbefehle zum Bot hinzu."""
        # Basiskommandos
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        
        # Trading-Kommandos
        self.application.add_handler(CommandHandler("balance", self._balance_command))
        self.application.add_handler(CommandHandler("positions", self._positions_command))
        self.application.add_handler(CommandHandler("performance", self._performance_command))
        
        # Bot-Steuerungskommandos
        self.application.add_handler(CommandHandler("start_bot", self._start_bot_command))
        self.application.add_handler(CommandHandler("stop_bot", self._stop_bot_command))
        self.application.add_handler(CommandHandler("pause_bot", self._pause_bot_command))
        self.application.add_handler(CommandHandler("resume_bot", self._resume_bot_command))
        
        # Admin-Kommandos
        self.application.add_handler(CommandHandler("restart", self._restart_command))
        
        # Transkript-Verarbeitung
        self.application.add_handler(CommandHandler("process_transcript", self._process_transcript_command))
        
        # Handler für normale Nachrichten (für Transkript-Aufzeichnung)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
        
        self.logger.info("Standard-Befehlshandler hinzugefügt")
    
    def register_commands(self, command_handlers: Dict[str, Callable]):
        """
        Registriert zusätzliche Befehle für den MainController.
        
        Args:
            command_handlers: Dictionary mit Befehlsnamen und zugehörigen Funktionen
        """
        try:
            self.command_handlers = command_handlers
            self.logger.info(f"{len(command_handlers)} Befehle vom MainController registriert")
        except Exception as e:
            self.logger.error(f"Fehler beim Registrieren von Befehlen: {str(e)}")
    
    def start(self):
        """Startet den Telegram-Bot in einem separaten Thread."""
        if not self.is_configured:
            self.logger.error("Telegram-Bot ist nicht konfiguriert und kann nicht gestartet werden")
            return False
        
        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return False
        
        try:
            # Thread für Bot-Polling starten
            self.bot_thread = threading.Thread(target=self._run_bot)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            self.is_running = True
            self.logger.info("Telegram-Bot erfolgreich gestartet")
            
            # Sende Startup-Nachricht an alle erlaubten Benutzer
            startup_message = "🤖 Trading Bot wurde gestartet und ist bereit für Befehle.\nVerwende /help für eine Liste der verfügbaren Befehle."
            self._broadcast_message(startup_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram-Bots: {str(e)}")
            return False
    
    def stop(self):
        """Stoppt den Telegram-Bot."""
        if not self.is_running:
            self.logger.warning("Telegram-Bot läuft nicht")
            return False
        
        try:
            # Bot-Polling beenden
            if self.application:
                self.application.stop()
            
            self.is_running = False
            
            # Warten, bis der Thread beendet ist
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=10)
            
            self.logger.info("Telegram-Bot erfolgreich gestoppt")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Telegram-Bots: {str(e)}")
            return False
    
    def _run_bot(self):
        """Führt den Bot-Polling-Loop aus."""
        try:
            self.logger.info("Starte Telegram-Bot-Polling...")
            
            # Polling starten
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            self.logger.error(f"Fehler im Bot-Polling-Loop: {str(e)}")
            self.is_running = False
    
    def _check_authorized(self, update: Update) -> bool:
        """
        Prüft, ob der Benutzer autorisiert ist, den Bot zu verwenden.
        
        Args:
            update: Telegram-Update-Objekt
            
        Returns:
            True, wenn autorisiert, sonst False
        """
        if not self.allowed_users:
            self.logger.warning("Befehl erhalten, aber keine autorisierten Benutzer konfiguriert")
            return False
        
        user_id = update.effective_user.id
        
        if user_id in self.allowed_users:
            return True
        
        self.logger.warning(f"Nicht autorisierter Zugriff von Benutzer {user_id} ({update.effective_user.username})")
        return False
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /start Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        user_name = update.effective_user.first_name
        
        welcome_message = (
            f"👋 Hallo {user_name}!\n\n"
            f"Willkommen beim Gemma Trading Bot. "
            f"Ich bin dein Assistent für das Überwachen und Steuern des Trading-Bots.\n\n"
            f"Verwende /help, um eine Liste der verfügbaren Befehle zu sehen."
        )
        
        await update.message.reply_text(welcome_message)
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /help Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        help_text = (
            "🤖 *Gemma Trading Bot - Hilfe*\n\n"
            "*Basis-Befehle:*\n"
            "/start - Startet den Bot\n"
            "/help - Zeigt diese Hilfe an\n"
            "/status - Zeigt den aktuellen Status des Trading-Bots\n\n"
            
            "*Trading-Informationen:*\n"
            "/balance - Zeigt den aktuellen Kontostand\n"
            "/positions - Zeigt offene Positionen\n"
            "/performance - Zeigt Performance-Metriken\n\n"
            
            "*Bot-Steuerung:*\n"
            "/start_bot - Startet den Trading-Bot\n"
            "/stop_bot - Stoppt den Trading-Bot\n"
            "/pause_bot - Pausiert den Trading-Bot\n"
            "/resume_bot - Setzt den Trading-Bot fort\n\n"
            
            "*Admin-Befehle:*\n"
            "/restart - Startet den Trading-Bot neu\n"
            "/process_transcript - Verarbeitet ein Transkript"
        )
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /status Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'get_status'):
                status = self.main_controller.get_status()
                
                # Formatierte Statusnachricht erstellen
                message = (
                    f"📊 *Trading Bot Status*\n\n"
                    f"🔄 *Status*: {status.get('state', 'Unbekannt')}\n"
                    f"🔴 *Notfallmodus*: {'Aktiv' if status.get('emergency_mode', False) else 'Inaktiv'}\n"
                    f"⏱ *Uptime*: {status.get('uptime', 'Unbekannt')}\n\n"
                    
                    f"📦 *Module*:\n"
                )
                
                # Module-Status
                for module, module_status in status.get('modules', {}).items():
                    status_emoji = "✅" if module_status.get('status') == "running" else "⏸" if module_status.get('status') == "paused" else "⛔"
                    message += f"  {status_emoji} {module}: {module_status.get('status', 'Unbekannt')}\n"
                
                # Letzte Ereignisse
                events = status.get('events', [])
                if events:
                    message += "\n🔍 *Letzte Ereignisse*:\n"
                    for event in events[:5]:  # Nur die letzten 5 Ereignisse
                        event_time = datetime.fromisoformat(event.get('timestamp', '')).strftime('%H:%M:%S')
                        message += f"  • {event_time} - {event.get('type', 'Unbekannt')}: {event.get('title', 'Kein Titel')}\n"
                
                await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text("⚠️ Kann Status nicht abrufen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Status-Abruf: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen des Status: {str(e)}")
    
    async def _balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /balance Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, '_get_account_balance'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Rufe Kontostand ab...")
                
                # Kontostand abrufen
                balance_data = self.main_controller._get_account_balance()
                
                if balance_data.get('status') == 'success':
                    balance = balance_data.get('balance', {})
                    
                    # Nachricht erstellen
                    message = "💰 *Kontostand*\n\n"
                    
                    if 'total' in balance:
                        message += "*Gesamt:*\n"
                        for currency, amount in balance['total'].items():
                            if float(amount) > 0:
                                message += f"  • {currency}: {amount}\n"
                    
                    if 'free' in balance:
                        message += "\n*Verfügbar:*\n"
                        for currency, amount in balance['free'].items():
                            if float(amount) > 0:
                                message += f"  • {currency}: {amount}\n"
                    
                    if 'used' in balance:
                        message += "\n*In Verwendung:*\n"
                        for currency, amount in balance['used'].items():
                            if float(amount) > 0:
                                message += f"  • {currency}: {amount}\n"
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(f"❌ Fehler beim Abrufen des Kontostands: {balance_data.get('message', 'Unbekannter Fehler')}")
            else:
                await update.message.reply_text("⚠️ Kann Kontostand nicht abrufen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Kontostand-Abruf: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen des Kontostands: {str(e)}")
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /positions Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, '_get_open_positions'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Rufe offene Positionen ab...")
                
                # Positionen abrufen
                positions_data = self.main_controller._get_open_positions()
                
                if positions_data.get('status') == 'success':
                    positions = positions_data.get('positions', [])
                    
                    if not positions:
                        await status_message.edit_text("📊 Keine offenen Positionen vorhanden")
                        return
                    
                    # Nachricht erstellen
                    message = "📊 *Offene Positionen*\n\n"
                    
                    for pos in positions:
                        symbol = pos.get('symbol', 'Unbekannt')
                        side = pos.get('side', 'Unbekannt')
                        size = pos.get('contracts', 0)
                        entry_price = pos.get('entryPrice', 0)
                        current_price = pos.get('markPrice', 0)
                        unrealized_pnl = pos.get('unrealizedPnl', 0)
                        leverage = pos.get('leverage', 1)
                        
                        # PnL in Prozent berechnen
                        if float(entry_price) > 0:
                            if side == 'long':
                                pnl_percent = (float(current_price) / float(entry_price) - 1) * 100
                            else:
                                pnl_percent = (1 - float(current_price) / float(entry_price)) * 100
                        else:
                            pnl_percent = 0
                        
                        # Emojis basierend auf PnL
                        if pnl_percent > 0:
                            emoji = "🟢"
                        elif pnl_percent < 0:
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                        
                        # Side formatieren
                        side_formatted = "LONG 📈" if side == 'long' else "SHORT 📉" if side == 'short' else side
                        
                        message += (
                            f"{emoji} *{symbol}* ({side_formatted})\n"
                            f"  • Größe: {size} Kontrakte (Hebel: {leverage}x)\n"
                            f"  • Einstieg: {entry_price}\n"
                            f"  • Aktuell: {current_price}\n"
                            f"  • PnL: {unrealized_pnl} ({pnl_percent:.2f}%)\n\n"
                        )
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(f"❌ Fehler beim Abrufen der Positionen: {positions_data.get('message', 'Unbekannter Fehler')}")
            else:
                await update.message.reply_text("⚠️ Kann Positionen nicht abrufen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Positionen-Abruf: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen der Positionen: {str(e)}")
    
    async def _performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /performance Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, '_get_performance_metrics'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Rufe Performance-Metriken ab...")
                
                # Metriken abrufen
                metrics_data = self.main_controller._get_performance_metrics()
                
                if metrics_data.get('status') == 'success':
                    metrics = metrics_data.get('metrics', {})
                    
                    # Nachricht erstellen
                    message = "📈 *Performance-Metriken*\n\n"
                    
                    # Trading-Metriken
                    if 'trading' in metrics:
                        trading = metrics['trading']
                        win_rate = trading.get('win_rate', 0) * 100
                        
                        message += "🎯 *Trading Performance*:\n"
                        message += f"  • Trades: {trading.get('total_trades', 0)}\n"
                        message += f"  • Gewonnen: {trading.get('winning_trades', 0)}\n"
                        message += f"  • Verloren: {trading.get('losing_trades', 0)}\n"
                        message += f"  • Gewinnrate: {win_rate:.2f}%\n"
                        message += f"  • Durchschn. Gewinn: {(trading.get('avg_win', 0) * 100):.2f}%\n"
                        message += f"  • Durchschn. Verlust: {(trading.get('avg_loss', 0) * 100):.2f}%\n"
                        message += f"  • Gesamt-PnL: {(trading.get('total_pnl', 0) * 100):.2f}%\n\n"
                    
                    # Learning-Metriken
                    if 'learning' in metrics:
                        learning = metrics['learning']
                        message += "🧠 *Learning Metrics*:\n"
                        
                        for key, value in learning.items():
                            message += f"  • {key}: {value}\n"
                        
                        message += "\n"
                    
                    # Steuer-Informationen
                    if 'tax' in metrics:
                        tax = metrics['tax']
                        message += "💸 *Steuerinformationen*:\n"
                        
                        if 'total_profit' in tax:
                            message += f"  • Gesamtgewinn: {tax['total_profit']}\n"
                        
                        if 'taxable_amount' in tax:
                            message += f"  • Steuerpflichtiger Betrag: {tax['taxable_amount']}\n"
                        
                        if 'tax_rate' in tax:
                            message += f"  • Steuersatz: {tax['tax_rate']*100}%\n"
                        
                        if 'estimated_tax' in tax:
                            message += f"  • Geschätzte Steuer: {tax['estimated_tax']}\n"
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(f"❌ Fehler beim Abrufen der Performance-Metriken: {metrics_data.get('message', 'Unbekannter Fehler')}")
            else:
                await update.message.reply_text("⚠️ Kann Performance-Metriken nicht abrufen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Performance-Abruf: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen der Performance-Metriken: {str(e)}")
    
    async def _start_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /start_bot Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'start'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Starte Trading Bot...")
                
                # Auto-Trading-Parameter aus Nachricht extrahieren
                auto_trade = True
                if context.args and len(context.args) > 0:
                    arg = context.args[0].lower()
                    if arg in ["false", "no", "0", "off"]:
                        auto_trade = False
                
                # Bot starten
                success = self.main_controller.start(auto_trade=auto_trade)
                
                if success:
                    await status_message.edit_text(
                        f"✅ Trading Bot erfolgreich gestartet!\n"
                        f"Auto-Trading: {'Aktiviert' if auto_trade else 'Deaktiviert'}"
                    )
                else:
                    await status_message.edit_text("❌ Fehler beim Starten des Trading Bots")
            else:
                await update.message.reply_text("⚠️ Kann Bot nicht starten - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Start: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Starten des Bots: {str(e)}")
    
    async def _stop_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /stop_bot Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'stop'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Stoppe Trading Bot...")
                
                # Bot stoppen
                success = self.main_controller.stop()
                
                if success:
                    await status_message.edit_text("✅ Trading Bot erfolgreich gestoppt!")
                else:
                    await status_message.edit_text("❌ Fehler beim Stoppen des Trading Bots")
            else:
                await update.message.reply_text("⚠️ Kann Bot nicht stoppen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Stopp: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Stoppen des Bots: {str(e)}")
    
    async def _pause_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /pause_bot Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'pause'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Pausiere Trading Bot...")
                
                # Bot pausieren
                success = self.main_controller.pause()
                
                if success:
                    await status_message.edit_text("⏸ Trading Bot erfolgreich pausiert!")
                else:
                    await status_message.edit_text("❌ Fehler beim Pausieren des Trading Bots")
            else:
                await update.message.reply_text("⚠️ Kann Bot nicht pausieren - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Pause: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Pausieren des Bots: {str(e)}")
    
    async def _resume_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /resume_bot Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'resume'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Setze Trading Bot fort...")
                
                # Bot fortsetzen
                success = self.main_controller.resume()
                
                if success:
                    await status_message.edit_text("▶️ Trading Bot erfolgreich fortgesetzt!")
                else:
                    await status_message.edit_text("❌ Fehler beim Fortsetzen des Trading Bots")
            else:
                await update.message.reply_text("⚠️ Kann Bot nicht fortsetzen - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Fortsetzung: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Fortsetzen des Bots: {str(e)}")
    
    async def _restart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /restart Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            if self.main_controller and hasattr(self.main_controller, 'restart'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text("🔄 Starte Trading Bot neu...")
                
                # Bot neu starten
                success = self.main_controller.restart()
                
                if success:
                    await status_message.edit_text("✅ Trading Bot erfolgreich neu gestartet!")
                else:
                    await status_message.edit_text("❌ Fehler beim Neustarten des Trading Bots")
            else:
                await update.message.reply_text("⚠️ Kann Bot nicht neu starten - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Neustart: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Neustarten des Bots: {str(e)}")
    
    async def _process_transcript_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /process_transcript Befehl."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            # Prüfen, ob ein Transkript-Pfad angegeben wurde
            if not context.args or len(context.args) == 0:
                recent_transcripts = self._get_recent_transcripts()
                
                if not recent_transcripts:
                    await update.message.reply_text(
                        "⚠️ Bitte gib einen Transkript-Pfad an oder zeichne erst ein Transkript auf.\n"
                        "Beispiel: /process_transcript data/transcripts/transcript_20230101.txt"
                    )
                    return
                
                # Verwende das neueste Transkript
                transcript_path = recent_transcripts[0]['path']
                await update.message.reply_text(
                    f"ℹ️ Verwende das neueste Transkript: {transcript_path}\n"
                    f"Starte Verarbeitung..."
                )
            else:
                transcript_path = context.args[0]
            
            if self.main_controller and hasattr(self.main_controller, '_process_transcript'):
                # Status-Nachricht senden
                status_message = await update.message.reply_text(f"🔄 Verarbeite Transkript: {transcript_path}...")
                
                # Transkript verarbeiten
                params = {'path': transcript_path}
                result = self.main_controller._process_transcript_command(params)
                
                if result.get('status') == 'success':
                    await status_message.edit_text(
                        f"✅ Transkript erfolgreich verarbeitet!\n\n"
                        f"Datei: {transcript_path}\n"
                        f"Ergebnis: {json.dumps(result.get('result', {}), indent=2)}"
                    )
                else:
                    await status_message.edit_text(
                        f"❌ Fehler bei der Transkript-Verarbeitung: {result.get('message', 'Unbekannter Fehler')}"
                    )
            else:
                await update.message.reply_text("⚠️ Kann Transkript nicht verarbeiten - MainController nicht verfügbar")
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            await update.message.reply_text(f"❌ Fehler bei der Transkript-Verarbeitung: {str(e)}")
    
    def _get_recent_transcripts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste der zuletzt aufgezeichneten Transkripte zurück.
        
        Args:
            limit: Maximale Anzahl der zurückzugebenden Transkripte
            
        Returns:
            Liste mit Transkript-Informationen
        """
        try:
            transcripts = []
            
            # Suche nach Transkriptdateien
            for file_path in sorted(self.transcript_dir.glob('transcript_*.txt'), reverse=True):
                if len(transcripts) >= limit:
                    break
                
                # Metadaten sammeln
                stats = file_path.stat()
                
                transcripts.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': stats.st_size,
                    'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    'created': datetime.fromtimestamp(stats.st_ctime).isoformat()
                })
            
            return transcripts
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Transkripte: {str(e)}")
            return []
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für normale Nachrichten (für Transkript-Aufzeichnung)."""
        if not self._check_authorized(update):
            return
        
        # Prüfen, ob eine Transkript-Aufzeichnung aktiv ist
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Hier könnte eine Implementierung zur Transkript-Aufzeichnung erfolgen
        # Beispiel: Speichern der Nachricht in einer Transkriptdatei
    
    def send_notification(self, title: str, message: str, priority: str = 'normal'):
        """
        Sendet eine Benachrichtigung an alle autorisierten Benutzer.
        
        Args:
            title: Titel der Benachrichtigung
            message: Nachrichtentext
            priority: Priorität ('low', 'normal', 'high', 'critical')
        """
        if not self.is_configured or not self.is_running or not self.allowed_users:
            self.logger.warning("Kann keine Benachrichtigung senden: Bot nicht konfiguriert/gestartet oder keine autorisierten Benutzer")
            return
        
        # Prüfe, ob wir die Benachrichtigung senden sollten (basierend auf Priorität und Cooldown)
        if not self._should_send_notification(priority):
            self.logger.debug(f"Benachrichtigung unterdrückt (Priorität: {priority})")
            return
        
        # Formatiere die Nachricht mit Prioritäts-Emoji
        priority_emoji = self._get_priority_emoji(priority)
        formatted_message = f"{priority_emoji} *{title}*\n\n{message}"
        
        # Sende die Nachricht an alle autorisierten Benutzer
        for user_id in self.allowed_users:
            try:
                if self.bot:
                    self.bot.send_message(
                        chat_id=user_id,
                        text=formatted_message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    self.logger.debug(f"Benachrichtigung an Benutzer {user_id} gesendet")
                else:
                    self.logger.error("Bot-Instanz nicht verfügbar")
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Benachrichtigung an Benutzer {user_id}: {str(e)}")
        
        # Aktualisiere Zähler und Zeitstempel
        self._update_notification_stats(priority)
    
    def _should_send_notification(self, priority: str) -> bool:
        """
        Prüft, ob eine Benachrichtigung basierend auf Priorität und Cooldown gesendet werden sollte.
        
        Args:
            priority: Priorität der Benachrichtigung
            
        Returns:
            True, wenn die Benachrichtigung gesendet werden sollte, sonst False
        """
        now = datetime.now()
        
        # Prüfe, ob die stündliche Zurücksetzung fällig ist
        if now >= self.notification_reset_time:
            # Zurücksetzen der Zähler
            for p in self.notification_counts.keys():
                self.notification_counts[p] = 0
            
            # Neue Reset-Zeit setzen
            self.notification_reset_time = now + timedelta(hours=1)
        
        # Prüfe, ob wir das stündliche Limit überschritten haben
        max_per_hour = self.max_notifications_per_hour.get(priority, 10)
        if self.notification_counts.get(priority, 0) >= max_per_hour:
            return False
        
        # Prüfe, ob die Abklingzeit noch aktiv ist
        last_time = self.last_notification_time.get(priority)
        if last_time and (now - last_time).total_seconds() < self.notification_cooldown:
            # Nur niedrigere Prioritäten unterdrücken
            if priority in ['low', 'normal']:
                return False
        
        return True
    
    def _update_notification_stats(self, priority: str):
        """
        Aktualisiert die Benachrichtigungsstatistiken.
        
        Args:
            priority: Priorität der gesendeten Benachrichtigung
        """
        now = datetime.now()
        
        # Zeitstempel aktualisieren
        self.last_notification_time[priority] = now
        
        # Zähler erhöhen
        self.notification_counts[priority] = self.notification_counts.get(priority, 0) + 1
    
    def _get_priority_emoji(self, priority: str) -> str:
        """
        Gibt das Emoji für die angegebene Priorität zurück.
        
        Args:
            priority: Priorität ('low', 'normal', 'high', 'critical')
            
        Returns:
            Emoji für die Priorität
        """
        if priority == 'critical':
            return "🚨"
        elif priority == 'high':
            return "⚠️"
        elif priority == 'normal':
            return "ℹ️"
        else:  # low
            return "📌"
    
    def _broadcast_message(self, message: str):
        """
        Sendet eine Nachricht an alle autorisierten Benutzer.
        
        Args:
            message: Nachrichtentext
        """
        if not self.is_configured or not self.allowed_users:
            return
        
        for user_id in self.allowed_users:
            try:
                if self.bot:
                    self.bot.send_message(chat_id=user_id, text=message)
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Broadcast-Nachricht an Benutzer {user_id}: {str(e)}")

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'allowed_users': [int(x) for x in os.getenv('TELEGRAM_ALLOWED_USERS', '').split(',') if x],
        'notification_level': 'INFO',
        'status_update_interval': 3600
    }
    
    # Telegram-Schnittstelle initialisieren
    telegram_interface = TelegramInterface(config)
    
    # Bot starten
    if telegram_interface.is_configured:
        telegram_interface.start()
        
        # Beispiel-Benachrichtigung senden
        telegram_interface.send_notification(
            "Test-Benachrichtigung",
            "Dies ist eine Test-Benachrichtigung vom Trading Bot.",
            priority="normal"
        )

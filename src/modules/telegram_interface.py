# telegram_interface.py

import os
import logging
import threading
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import io
from PIL import Image
import os
os.environ['PYTHONUNBUFFERED'] = '1'  # Deaktiviert Signalhandling im Thread

# Innerhalb der TelegramInterface-Klasse:
# In telegram_interface.py

def _run_bot(self):
    """Führt den Telegram-Bot komplett ohne asyncio und Signal-Handler aus"""
    try:
        import requests
        self.session = requests.Session()
        last_update_id = 0
        self.logger.info("Bot-Thread gestartet (HTTP-Polling-Modus)")
        while self.is_running:
            try:
                # Direkter API-Aufruf
                response = self.session.get(
                    f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                    params={
                        "offset": last_update_id + 1,
                        "timeout": 30,
                        "allowed_updates": ["message", "callback_query"]
                    },
                    timeout=35
                )
                response.raise_for_status()
                # Verarbeite Updates
                data = response.json()
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        last_update_id = update["update_id"]
                        self._handle_raw_update(update)
            except Exception as e:
                self.logger.error(f"Polling-Fehler: {str(e)}")
                time.sleep(5)
    except Exception as e:
        self.logger.error(f"Kritischer Bot-Fehler: {str(e)}")
        self.logger.error(traceback.format_exc())
    finally:
        self.logger.info("Bot-Thread beendet")
        self.is_running = False

def _handle_raw_update(self, update):
    """Verarbeitet Roh-Updates der Telegram API"""
    try:
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = message.get("text", "")
        
        # Autorisierung prüfen
        if self.allowed_users and user_id not in self.allowed_users:
            return
            
        # Befehle verarbeiten
        if text.startswith("/"):
            command = text.split()[0].lower()
            if command == "/start":
                self._send_direct_message(chat_id, "🤖 Bot aktiv! Nutze /help für Befehle")
            elif command == "/help":
                self._send_direct_message(chat_id, "📋 Verfügbare Befehle:\n/status - Bot-Status\n/balance - Kontostand")
            # Weitere Befehle hier ergänzen
            
    except Exception as e:
        self.logger.error(f"Update-Verarbeitungsfehler: {str(e)}")

def _send_direct_message(self, chat_id, text):
    """Sendet Nachricht direkt über die Telegram API"""
    try:
        self.session.post(
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
        )
    except Exception as e:
        self.logger.error(f"Nachricht konnte nicht gesendet werden: {str(e)}")
        
        # Starte den minimalen Polling-Loop
        # HIER IST DIE WICHTIGE ÄNDERUNG:
        self.loop.run_until_complete(minimal_polling_loop())
    except Exception as e:
        self.logger.error(f"Fehler im Bot-Thread: {str(e)}")
        self.logger.error(traceback.format_exc())
    finally:
        self.logger.info("Bot-Thread beendet")
        self.is_running = False

# Telegram-Bibliotheken
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes
from telegram.ext import filters
from telegram.constants import ParseMode

# Für Headless-Server (ohne GUI)
matplotlib.use('Agg')

class TelegramInterface:
    """
    Telegram-Bot-Schnittstelle für die Fernsteuerung und Benachrichtigungen des Trading-Bots.
    Ermöglicht die Interaktion mit dem Bot über Telegram-Nachrichten und bietet ein
    benutzerfreundliches Dashboard mit Trading-Informationen und Steuerungselementen.
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
        self.loop = None
        
        # Thread für Bot-Updates
        self.bot_thread = None
        self.is_running = False
        
        # Befehlsreferenzen (für dynamische Befehle)
        self.command_handlers = {}
        self.custom_commands = {}
        
        # Verzeichnis für aufgezeichnete Transkripte
        self.transcript_dir = Path('data/transcripts')
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Verzeichnis für temporäre Grafiken
        self.charts_dir = Path('data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Benutzer-Zustände (für mehrstufige Interaktionen)
        self.user_states = {}
        
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
            self.logger.error(traceback.format_exc())
            self.is_configured = False
    
    def _add_default_handlers(self):
        """Fügt die Standard-Befehlshandler zur Application hinzu."""
        try:
            # Basis-Befehle
            self.application.add_handler(CommandHandler("start", self._cmd_start))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            
            # Trading-Befehle
            self.application.add_handler(CommandHandler("startbot", self._cmd_startbot))
            self.application.add_handler(CommandHandler("stopbot", self._cmd_stopbot))
            self.application.add_handler(CommandHandler("pausebot", self._cmd_pausebot))
            self.application.add_handler(CommandHandler("resumebot", self._cmd_resumebot))
            
            # Informationsbefehle
            self.application.add_handler(CommandHandler("balance", self._cmd_balance))
            self.application.add_handler(CommandHandler("positions", self._cmd_positions))
            self.application.add_handler(CommandHandler("performance", self._cmd_performance))
            
            # Transkriptverarbeitung
            self.application.add_handler(CommandHandler("processtranscript", self._cmd_process_transcript))
            self.application.add_handler(MessageHandler(filters.Document.TEXT, self._handle_transcript_file))
            
            # Callback-Handler für Inline-Buttons
            self.application.add_handler(CallbackQueryHandler(self._handle_callback))
            
            # Unbekannte Befehle und Nachrichten
            self.application.add_handler(MessageHandler(filters.COMMAND, self._cmd_unknown))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
            
            # Error-Handler
            self.application.add_error_handler(self._error_handler)
            
            self.logger.info("Standard-Befehlshandler erfolgreich hinzugefügt")
        except Exception as e:
            self.logger.error(f"Fehler beim Hinzufügen der Standard-Befehlshandler: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def register_commands(self, commands: Dict[str, Callable]):
        """
        Registriert benutzerdefinierte Befehle vom MainController.
        
        Args:
            commands: Dictionary mit Befehlsnamen als Schlüssel und Callback-Funktionen als Werte
        """
        try:
            self.logger.info(f"Registriere {len(commands)} benutzerdefinierte Befehle")
            self.custom_commands = commands
            
            # Prüfe, ob Bot bereits läuft, dann müssen wir die Handler nicht sofort hinzufügen
            if not self.is_running:
                self.logger.debug("Bot läuft noch nicht, Handler werden bei Start hinzugefügt")
                return
            
            self.logger.info("Benutzerdefinierte Befehle erfolgreich registriert")
        except Exception as e:
            self.logger.error(f"Fehler beim Registrieren benutzerdefinierter Befehle: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def start(self):
        """Startet den Telegram-Bot in einem separaten Thread."""
        if not self.is_configured:
            self.logger.warning("Telegram-Bot nicht konfiguriert, kann nicht gestartet werden")
            return False
        
        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return True
        
        try:
            # Neuen Event-Loop erstellen und im Thread verwenden
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            
            # Kurz warten, um sicherzustellen, dass der Bot gestartet wird
            time.sleep(1)
            
            self.is_running = True
            self.logger.info("Telegram-Bot gestartet")
            
            # Initialen Status an alle Benutzer senden
            self._send_status_to_all_users("Bot gestartet", "Der Trading Bot wurde erfolgreich gestartet und ist bereit für Befehle.")
            
            # Timer für regelmäßige Statusupdates starten
            if self.status_update_interval > 0:
                self._start_status_update_timer()
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_bot(self):
        """Führt den Telegram-Bot im Hintergrund aus."""
        try:
            # Neuen Event-Loop erstellen und setzen
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Bot starten und auf Updates warten
            self.loop.run_until_complete(self.application.run_polling(allowed_updates=Update.ALL_TYPES))
        except Exception as e:
            self.logger.error(f"Fehler im Bot-Thread: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Bot-Thread beendet")
            self.is_running = False
    
    def stop(self):
        """Stoppt den Telegram-Bot."""
        if not self.is_running:
            self.logger.warning("Telegram-Bot läuft nicht")
            return True
        
        try:
            # Bot stoppen
            if self.loop and self.application:
                asyncio.run_coroutine_threadsafe(self.application.stop(), self.loop)
            
            # Warten, bis der Thread beendet ist
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=10)
            
            self.is_running = False
            self.logger.info("Telegram-Bot gestoppt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sendet eine Benachrichtigung an alle erlaubten Benutzer.
        
        Args:
            title: Titel der Benachrichtigung
            message: Inhalt der Benachrichtigung
            priority: Priorität ('low', 'normal', 'high', 'critical')
        """
        if not self.is_running or not self.allowed_users:
            return
        
        # Prüfe Benachrichtigungslimits
        if not self._check_notification_limits(priority):
            return
        
        # Formatiere die Nachricht je nach Priorität
        formatted_message = self._format_notification(title, message, priority)
        
        # Sende an alle erlaubten Benutzer
        for user_id in self.allowed_users:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_message(user_id, formatted_message, parse_mode=ParseMode.HTML),
                    self.loop
                )
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Benachrichtigung an {user_id}: {str(e)}")
    
    def _check_notification_limits(self, priority: str) -> bool:
        """
        Prüft, ob Benachrichtigungslimits erreicht wurden.
        
        Args:
            priority: Priorität der Benachrichtigung
        
        Returns:
            True, wenn die Benachrichtigung gesendet werden darf, sonst False
        """
        # Zurücksetzen der Zähler nach einer Stunde
        current_time = datetime.now()
        if current_time > self.notification_reset_time:
            self.notification_counts = {k: 0 for k in self.notification_counts}
            self.notification_reset_time = current_time + timedelta(hours=1)
        
        # Prüfe, ob das stündliche Limit erreicht wurde
        if self.notification_counts[priority] >= self.max_notifications_per_hour[priority]:
            self.logger.warning(f"Stündliches Limit für {priority}-Priorität erreicht")
            return False
        
        # Prüfe Cooldown für diese Priorität
        if priority in self.last_notification_time:
            time_since_last = (current_time - self.last_notification_time[priority]).total_seconds()
            if time_since_last < self.notification_cooldown:
                self.logger.debug(f"Cooldown für {priority}-Priorität aktiv ({time_since_last:.1f}s/{self.notification_cooldown}s)")
                return False
        
        # Aktualisiere Zähler und Zeitstempel
        self.notification_counts[priority] += 1
        self.last_notification_time[priority] = current_time
        return True
    
    def _format_notification(self, title: str, message: str, priority: str) -> str:
        """
        Formatiert eine Benachrichtigung basierend auf der Priorität.
        
        Args:
            title: Titel der Benachrichtigung
            message: Inhalt der Benachrichtigung
            priority: Priorität ('low', 'normal', 'high', 'critical')
        
        Returns:
            Formatierte Nachricht als HTML-String
        """
        # Emoji basierend auf Priorität
        emoji_map = {
            'low': '📝',
            'normal': 'ℹ️',
            'high': '⚠️',
            'critical': '🚨'
        }
        emoji = emoji_map.get(priority, 'ℹ️')
        
        # HTML-Formatierung
        if priority == 'critical':
            formatted_title = f"<b>{emoji} {title.upper()} {emoji}</b>"
        elif priority == 'high':
            formatted_title = f"<b>{emoji} {title}</b>"
        else:
            formatted_title = f"<b>{emoji} {title}</b>"
        
        return f"{formatted_title}\n\n{message}"
    
    async def _send_message(self, chat_id: int, text: str, parse_mode=None, reply_markup=None):
        """
        Sendet eine Nachricht an einen Chat.
        
        Args:
            chat_id: Telegram Chat-ID
            text: Nachrichtentext
            parse_mode: Optionaler Parse-Modus (HTML, Markdown)
            reply_markup: Optionale Antwort-Markup (InlineKeyboard)
        """
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Nachricht an {chat_id}: {str(e)}")
    
    def _start_status_update_timer(self):
        """Startet einen Timer für regelmäßige Statusupdates."""
        def send_periodic_updates():
            while self.is_running:
                try:
                    # Status abrufen und senden
                    if self.main_controller:
                        status = self.main_controller.get_status()
                        self._send_status_summary_to_all_users(status)
                except Exception as e:
                    self.logger.error(f"Fehler beim Senden des regelmäßigen Statusupdates: {str(e)}")
                
                # Warten bis zum nächsten Update
                time.sleep(self.status_update_interval)
        
        threading.Thread(target=send_periodic_updates, daemon=True).start()
        self.logger.info(f"Status-Update-Timer gestartet (Intervall: {self.status_update_interval}s)")
    
    def _send_status_to_all_users(self, title: str, message: str):
        """
        Sendet eine Statusnachricht an alle erlaubten Benutzer.
        
        Args:
            title: Titel der Statusnachricht
            message: Inhalt der Statusnachricht
        """
        for user_id in self.allowed_users:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_message(user_id, f"<b>{title}</b>\n\n{message}", parse_mode=ParseMode.HTML),
                    self.loop
                )
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Statusnachricht an {user_id}: {str(e)}")
    
    def _send_status_summary_to_all_users(self, status: Dict[str, Any]):
        """
        Sendet eine Zusammenfassung des aktuellen Bot-Status an alle erlaubten Benutzer.
        
        Args:
            status: Status-Dictionary vom MainController
        """
        try:
            # Status formatieren
            bot_state = status.get('state', 'unbekannt')
            emoji_map = {
                'running': '🟢',
                'paused': '🟠',
                'emergency': '🔴',
                'error': '🔴',
                'stopped': '⚪',
                'ready': '🔵',
                'initializing': '🔵'
            }
            emoji = emoji_map.get(bot_state, '⚪')
            
            # Module-Status
            module_status = status.get('modules', {})
            active_modules = sum(1 for mod in module_status.values() if mod.get('status') == 'running')
            total_modules = len(module_status)
            
            # Letzte Events
            events = status.get('events', [])
            recent_events = events[-3:] if events else []
            
            message = (
                f"<b>Trading Bot Status</b> {emoji}\n\n"
                f"Status: <b>{bot_state.upper()}</b>\n"
                f"Aktive Module: {active_modules}/{total_modules}\n"
                f"Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
            )
            
            if recent_events:
                message += "<b>Letzte Ereignisse:</b>\n"
                for event in reversed(recent_events):
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    message += f"• {event_time} - {event['title']}\n"
            
            # Steuerungsbuttons hinzufügen
            keyboard = [
                [
                    InlineKeyboardButton("🟢 Start", callback_data="startbot"),
                    InlineKeyboardButton("🔴 Stop", callback_data="stopbot"),
                    InlineKeyboardButton("⏸️ Pause", callback_data="pausebot")
                ],
                [
                    InlineKeyboardButton("💰 Kontostand", callback_data="balance"),
                    InlineKeyboardButton("📊 Positionen", callback_data="positions"),
                    InlineKeyboardButton("📈 Performance", callback_data="performance")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # An alle Benutzer senden
            for user_id in self.allowed_users:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._send_message(
                            user_id, 
                            message, 
                            parse_mode=ParseMode.HTML,
                            reply_markup=reply_markup
                        ),
                        self.loop
                    )
                except Exception as e:
                    self.logger.error(f"Fehler beim Senden der Statuszusammenfassung an {user_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Statuszusammenfassung: {str(e)}")
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /start-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        welcome_message = (
            f"👋 Willkommen beim Trading Bot!\n\n"
            f"Ich bin deine Schnittstelle zum Gemma Trading Bot. "
            f"Du kannst mich verwenden, um den Bot zu steuern, Statusupdates zu erhalten "
            f"und Trading-Informationen abzurufen.\n\n"
            f"Verwende /help, um eine Liste aller verfügbaren Befehle zu sehen."
        )
        
        # Grundlegende Steuerungsbuttons
        keyboard = [
            [
                InlineKeyboardButton("ℹ️ Status", callback_data="status"),
                InlineKeyboardButton("🟢 Bot starten", callback_data="startbot")
            ],
            [
                InlineKeyboardButton("📋 Befehle", callback_data="help"),
                InlineKeyboardButton("📊 Dashboard", callback_data="dashboard")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /help-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        help_message = (
            "<b>📋 Verfügbare Befehle:</b>\n\n"
            "<b>Grundlegende Befehle:</b>\n"
            "/start - Startet den Bot und zeigt das Willkommensmenü\n"
            "/help - Zeigt diese Hilfe an\n"
            "/status - Zeigt den aktuellen Status des Trading Bots\n\n"
            
            "<b>Trading-Steuerung:</b>\n"
            "/startbot - Startet den Trading Bot\n"
            "/stopbot - Stoppt den Trading Bot\n"
            "/pausebot - Pausiert den Trading Bot\n"
            "/resumebot - Setzt den pausierten Trading Bot fort\n\n"
            
            "<b>Trading-Informationen:</b>\n"
            "/balance - Zeigt den aktuellen Kontostand\n"
            "/positions - Zeigt offene Positionen\n"
            "/performance - Zeigt Performance-Metriken\n\n"
            
            "<b>Transkript-Verarbeitung:</b>\n"
            "/processtranscript [Pfad] - Verarbeitet ein Transkript\n"
            "Du kannst auch direkt Transkriptdateien (.txt) senden\n\n"
            
            "<b>Sonstige Funktionen:</b>\n"
            "/dashboard - Zeigt ein interaktives Dashboard"
        )
        
        await update.message.reply_text(help_message, parse_mode=ParseMode.HTML)
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /status-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        # Prüfen, ob MainController verfügbar ist
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        try:
            # Status vom MainController abrufen
            status = self.main_controller.get_status()
            
            # Status formatieren
            bot_state = status.get('state', 'unbekannt')
            emoji_map = {
                'running': '🟢',
                'paused': '🟠',
                'emergency': '🔴',
                'error': '🔴',
                'stopped': '⚪',
                'ready': '🔵',
                'initializing': '🔵'
            }
            emoji = emoji_map.get(bot_state, '⚪')
            
            # Detaillierte Statusinformationen
            module_status = status.get('modules', {})
            
            # Nachricht zusammenstellen
            message = (
                f"<b>Trading Bot Status</b> {emoji}\n\n"
                f"Status: <b>{bot_state.upper()}</b>\n"
                f"Version: {status.get('version', 'unbekannt')}\n"
                f"Laufzeit: {status.get('uptime', '00:00:00')}\n\n"
                f"<b>Module:</b>\n"
            )
            
            # Moduledetails hinzufügen
            for module_name, module_info in module_status.items():
                module_state = module_info.get('status', 'unbekannt')
                module_emoji = '🟢' if module_state == 'running' else '⚪'
                message += f"{module_emoji} {module_name}: {module_state}\n"
            
            # Letzte Events
            events = status.get('events', [])
            if events:
                message += "\n<b>Letzte Ereignisse:</b>\n"
                for event in events[-5:]:  # Zeige die letzten 5 Events
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    event_type = event['type']
                    event_title = event['title']
                    message += f"• {event_time} [{event_type}] {event_title}\n"
            
            # Steuerungsbuttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Aktualisieren", callback_data="refresh_status"),
                    InlineKeyboardButton("📊 Dashboard", callback_data="dashboard")
                ],
                [
                    InlineKeyboardButton("🟢 Start", callback_data="startbot"),
                    InlineKeyboardButton("🔴 Stop", callback_data="stopbot"),
                    InlineKeyboardButton("⏸️ Pause", callback_data="pausebot")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        except Exception as e:
            error_message = f"Fehler beim Abrufen des Status: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_startbot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /startbot-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Starte Trading Bot...")
        
        try:
            # Führe im Hintergrund aus, um Telegram-Thread nicht zu blockieren
            result = await self._run_controller_method(self.main_controller.start)
            
            if result:
                await update.message.reply_text("✅ Trading Bot erfolgreich gestartet")
            else:
                await update.message.reply_text("❌ Fehler beim Starten des Trading Bots")
        except Exception as e:
            error_message = f"Fehler beim Starten des Bots: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_stopbot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /stopbot-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Stoppe Trading Bot...")
        
        try:
            result = await self._run_controller_method(self.main_controller.stop)
            
            if result:
                await update.message.reply_text("✅ Trading Bot erfolgreich gestoppt")
            else:
                await update.message.reply_text("❌ Fehler beim Stoppen des Trading Bots")
        except Exception as e:
            error_message = f"Fehler beim Stoppen des Bots: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_pausebot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /pausebot-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Pausiere Trading Bot...")
        
        try:
            result = await self._run_controller_method(self.main_controller.pause)
            
            if result:
                await update.message.reply_text("✅ Trading Bot erfolgreich pausiert")
            else:
                await update.message.reply_text("❌ Fehler beim Pausieren des Trading Bots")
        except Exception as e:
            error_message = f"Fehler beim Pausieren des Bots: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_resumebot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /resumebot-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Setze Trading Bot fort...")
        
        try:
            result = await self._run_controller_method(self.main_controller.resume)
            
            if result:
                await update.message.reply_text("✅ Trading Bot erfolgreich fortgesetzt")
            else:
                await update.message.reply_text("❌ Fehler beim Fortsetzen des Trading Bots")
        except Exception as e:
            error_message = f"Fehler beim Fortsetzen des Bots: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /balance-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Rufe Kontostand ab...")
        
        try:
            result = await self._run_controller_method(self.main_controller._get_account_balance)
            
            if result.get('status') == 'success':
                balance_data = result.get('balance', {})
                message = "<b>💰 Kontostand</b>\n\n"
                
                # Je nach Format der Balance-Daten anpassen
                if isinstance(balance_data, dict):
                    for currency, amount in balance_data.items():
                        # Format Beträge mit angemessener Genauigkeit
                        if float(amount) < 0.0001:
                            formatted_amount = f"{float(amount):.8f}"
                        elif float(amount) < 0.01:
                            formatted_amount = f"{float(amount):.6f}"
                        else:
                            formatted_amount = f"{float(amount):.2f}"
                        
                        message += f"<b>{currency}</b>: {formatted_amount}\n"
                else:
                    message += str(balance_data)
                
                # Zeitstempel hinzufügen
                message += f"\n<i>Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
                
                await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                await update.message.reply_text(f"❌ Fehler: {error_msg}")
        except Exception as e:
            error_message = f"Fehler beim Abrufen des Kontostands: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /positions-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Rufe offene Positionen ab...")
        
        try:
            result = await self._run_controller_method(self.main_controller._get_open_positions)
            
            if result.get('status') == 'success':
                positions = result.get('positions', [])
                
                if not positions:
                    await update.message.reply_text("📊 Keine offenen Positionen vorhanden")
                    return
                
                message = "<b>📊 Offene Positionen</b>\n\n"
                
                for pos in positions:
                    symbol = pos.get('symbol', 'Unbekannt')
                    side = pos.get('side', 'Unbekannt')
                    size = pos.get('size', 0)
                    entry_price = pos.get('entry_price', 0)
                    current_price = pos.get('current_price', 0)
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_percent = pos.get('unrealized_pnl_percent', 0)
                    
                    # Emojis basierend auf Position
                    side_emoji = '🔴' if side.lower() == 'short' else '🟢'
                    pnl_emoji = '📈' if pnl >= 0 else '📉'
                    
                    message += (
                        f"{side_emoji} <b>{symbol}</b> ({side.upper()})\n"
                        f"Größe: {size}\n"
                        f"Einstieg: {entry_price}\n"
                        f"Aktuell: {current_price}\n"
                        f"PnL: {pnl_emoji} {pnl:.2f} ({pnl_percent:.2f}%)\n\n"
                    )
                
                # Zeitstempel hinzufügen
                message += f"<i>Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
                
                # Aktionsbuttons hinzufügen
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Aktualisieren", callback_data="refresh_positions"),
                        InlineKeyboardButton("❌ Alle schließen", callback_data="close_all_positions")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                await update.message.reply_text(f"❌ Fehler: {error_msg}")
        except Exception as e:
            error_message = f"Fehler beim Abrufen der Positionen: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /performance-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        await update.message.reply_text("Rufe Performance-Metriken ab...")
        
        try:
            result = await self._run_controller_method(self.main_controller._get_performance_metrics)
            
            if result.get('status') == 'success':
                metrics = result.get('metrics', {})
                
                message = "<b>📈 Performance-Metriken</b>\n\n"
                
                # Trading-Metriken
                if 'trading' in metrics:
                    trading = metrics['trading']
                    message += "<b>Trading-Performance:</b>\n"
                    message += f"Trades gesamt: {trading.get('total_trades', 0)}\n"
                    message += f"Gewinnende Trades: {trading.get('winning_trades', 0)}\n"
                    message += f"Verlierende Trades: {trading.get('losing_trades', 0)}\n"
                    
                    win_rate = trading.get('win_rate', 0) * 100
                    message += f"Win-Rate: {win_rate:.1f}%\n"
                    
                    avg_win = trading.get('avg_win', 0) * 100
                    avg_loss = trading.get('avg_loss', 0) * 100
                    message += f"Durchschn. Gewinn: {avg_win:.2f}%\n"
                    message += f"Durchschn. Verlust: {avg_loss:.2f}%\n"
                    
                    total_pnl = trading.get('total_pnl', 0) * 100
                    message += f"Gesamt-PnL: {total_pnl:.2f}%\n\n"
                
                # Steuerliche Informationen
                if 'tax' in metrics:
                    tax = metrics['tax']
                    message += "<b>Steuerliche Informationen:</b>\n"
                    message += f"Realisierte Gewinne: {tax.get('realized_gains', 0):.2f}\n"
                    message += f"Realisierte Verluste: {tax.get('realized_losses', 0):.2f}\n"
                    message += f"Netto-Gewinn: {tax.get('net_profit', 0):.2f}\n\n"
                
                # Lernmodul-Metriken
                if 'learning' in metrics:
                    learning = metrics['learning']
                    message += "<b>Modell-Performance:</b>\n"
                    message += f"Modell-Genauigkeit: {learning.get('model_accuracy', 0) * 100:.1f}%\n"
                    message += f"Backtest-Score: {learning.get('backtest_score', 0):.3f}\n"
                
                # Zeitstempel hinzufügen
                message += f"\n<i>Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
                
                # Performance-Visualisierung erstellen und senden
                if 'trading' in metrics and metrics['trading'].get('total_trades', 0) > 0:
                    # Generiere Visualisierung
                    chart_path = await self._create_performance_chart(metrics)
                    
                    # Text-Nachricht senden
                    await update.message.reply_text(message, parse_mode=ParseMode.HTML)
                    
                    # Bild separat senden
                    if chart_path:
                        with open(chart_path, 'rb') as photo:
                            await update.message.reply_photo(photo=photo)
                        
                        # Datei nach dem Senden löschen
                        os.remove(chart_path)
                    else:
                        await update.message.reply_text("Konnte keine Visualisierung erstellen")
                else:
                    # Nur Text-Nachricht senden, wenn keine Daten für Visualisierung
                    await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                await update.message.reply_text(f"❌ Fehler: {error_msg}")
        except Exception as e:
            error_message = f"Fehler beim Abrufen der Performance-Metriken: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _create_performance_chart(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Erstellt ein Leistungsdiagramm basierend auf den Metriken.
        
        Args:
            metrics: Performance-Metriken
        
        Returns:
            Pfad zur erstellten Bilddatei oder None bei Fehler
        """
        try:
            if 'trading' not in metrics or metrics['trading'].get('total_trades', 0) <= 0:
                return None
            
            trading = metrics['trading']
            
            # Datei für das Diagramm erstellen
            chart_filename = f"performance_{int(time.time())}.png"
            chart_path = self.charts_dir / chart_filename
            
            # Diagramm erstellen
            plt.figure(figsize=(10, 8))
            
            # 1. Pie-Chart für Gewinn/Verlust-Verhältnis
            plt.subplot(2, 1, 1)
            labels = ['Gewinnende Trades', 'Verlierende Trades']
            sizes = [trading.get('winning_trades', 0), trading.get('losing_trades', 0)]
            colors = ['#4CAF50', '#F44336']
            explode = (0.1, 0)  # Explode den ersten Slice (Gewinnende Trades)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=140)
            plt.axis('equal')  # Gleichmäßige Aspektverhältnisse für kreisförmigen Pie
            plt.title('Win/Loss-Verhältnis')
            
            # 2. Balkendiagramm für Durchschnittliche Gewinne/Verluste
            plt.subplot(2, 1, 2)
            categories = ['Durchschn. Gewinn', 'Durchschn. Verlust', 'Gesamt-PnL']
            values = [
                trading.get('avg_win', 0) * 100, 
                trading.get('avg_loss', 0) * 100, 
                trading.get('total_pnl', 0) * 100
            ]
            
            bars = plt.bar(categories, values, color=['#4CAF50', '#F44336', '#2196F3'])
            
            # Werte über den Balken anzeigen
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            plt.title('Durchschnittliche Gewinne und Verluste (%)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            
            return str(chart_path)
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Performance-Diagramms: {str(e)}")
            return None
    
    async def _cmd_process_transcript(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt den /processtranscript-Befehl."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        # Prüfen, ob ein Pfad angegeben wurde
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Bitte gib den Pfad zum Transkript an.\n"
                "Beispiel: /processtranscript data/transcripts/mein_transkript.txt\n\n"
                "Alternativ kannst du auch direkt eine Textdatei senden."
            )
            return
        
        transcript_path = context.args[0]
        
        await update.message.reply_text(f"Starte Verarbeitung des Transkripts: {transcript_path}...")
        
        try:
            result = await self._run_controller_method(
                self.main_controller._process_transcript_command,
                {'path': transcript_path}
            )
            
            if result.get('status') == 'success':
                result_data = result.get('result', {})
                
                # Erfolgreiche Verarbeitung
                message = (
                    f"✅ <b>Transkript erfolgreich verarbeitet</b>\n\n"
                    f"Datei: {result_data.get('file', 'Unbekannt')}\n"
                    f"Sprache: {result_data.get('language', 'Unbekannt')}\n"
                    f"Chunks: {result_data.get('chunks', 0)}\n"
                    f"Extrahierte Wissenselemente: {result_data.get('total_items', 0)}\n\n"
                )
                
                # Details zu Kategorien
                if 'knowledge_items' in result_data:
                    message += "<b>Elemente pro Kategorie:</b>\n"
                    for category, count in result_data['knowledge_items'].items():
                        readable_category = category.replace('_', ' ').title()
                        message += f"• {readable_category}: {count}\n"
                
                await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                await update.message.reply_text(f"❌ Fehler: {error_msg}")
        except Exception as e:
            error_message = f"Fehler bei der Verarbeitung des Transkripts: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _handle_transcript_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt hochgeladene Transkriptdateien."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        if not self.main_controller:
            await update.message.reply_text("Fehler: Kein Zugriff auf den MainController")
            return
        
        # Prüfen, ob es sich um eine Text-Datei handelt
        file = update.message.document
        if not file.file_name.endswith('.txt'):
            await update.message.reply_text(
                "❌ Ungültiger Dateityp. Bitte sende nur .txt-Dateien als Transkripte."
            )
            return
        
        await update.message.reply_text("Lade Transkriptdatei herunter...")
        
        try:
            # Datei herunterladen
            new_file = await context.bot.get_file(file.file_id)
            file_path = self.transcript_dir / file.file_name
            
            await new_file.download_to_drive(custom_path=file_path)
            
            await update.message.reply_text(f"Datei heruntergeladen: {file_path}\nStarte Verarbeitung...")
            
            # Transkript verarbeiten
            result = await self._run_controller_method(
                self.main_controller._process_transcript_command,
                {'path': str(file_path)}
            )
            
            if result.get('status') == 'success':
                result_data = result.get('result', {})
                
                # Erfolgreiche Verarbeitung
                message = (
                    f"✅ <b>Transkript erfolgreich verarbeitet</b>\n\n"
                    f"Datei: {file.file_name}\n"
                    f"Sprache: {result_data.get('language', 'Unbekannt')}\n"
                    f"Chunks: {result_data.get('chunks', 0)}\n"
                    f"Extrahierte Wissenselemente: {result_data.get('total_items', 0)}\n\n"
                )
                
                # Details zu Kategorien
                if 'knowledge_items' in result_data:
                    message += "<b>Elemente pro Kategorie:</b>\n"
                    for category, count in result_data['knowledge_items'].items():
                        readable_category = category.replace('_', ' ').title()
                        message += f"• {readable_category}: {count}\n"
                
                await update.message.reply_text(message, parse_mode=ParseMode.HTML)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                await update.message.reply_text(f"❌ Fehler: {error_msg}")
        except Exception as e:
            error_message = f"Fehler bei der Verarbeitung des Transkripts: {str(e)}"
            self.logger.error(error_message)
            await update.message.reply_text(error_message)
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt Callbacks von Inline-Buttons."""
        query = update.callback_query
        user_id = query.from_user.id
        
        if not self._is_authorized(user_id):
            await query.answer("Nicht autorisiert")
            return
        
        # Callback-Daten extrahieren
        callback_data = query.data
        
        try:
            # Button-Aktion bestätigen
            await query.answer()
            
            # Callback-Typ verarbeiten
            if callback_data == "startbot":
                # Bot starten
                message = await query.message.reply_text("Starte Trading Bot...")
                result = await self._run_controller_method(self.main_controller.start)
                
                if result:
                    await message.edit_text("✅ Trading Bot erfolgreich gestartet")
                else:
                    await message.edit_text("❌ Fehler beim Starten des Trading Bots")
            
            elif callback_data == "stopbot":
                # Bot stoppen
                message = await query.message.reply_text("Stoppe Trading Bot...")
                result = await self._run_controller_method(self.main_controller.stop)
                
                if result:
                    await message.edit_text("✅ Trading Bot erfolgreich gestoppt")
                else:
                    await message.edit_text("❌ Fehler beim Stoppen des Trading Bots")
            
            elif callback_data == "pausebot":
                # Bot pausieren
                message = await query.message.reply_text("Pausiere Trading Bot...")
                result = await self._run_controller_method(self.main_controller.pause)
                
                if result:
                    await message.edit_text("✅ Trading Bot erfolgreich pausiert")
                else:
                    await message.edit_text("❌ Fehler beim Pausieren des Trading Bots")
            
            elif callback_data == "resumebot":
                # Bot fortsetzen
                message = await query.message.reply_text("Setze Trading Bot fort...")
                result = await self._run_controller_method(self.main_controller.resume)
                
                if result:
                    await message.edit_text("✅ Trading Bot erfolgreich fortgesetzt")
                else:
                    await message.edit_text("❌ Fehler beim Fortsetzen des Trading Bots")
            
            elif callback_data == "status":
                # Status anzeigen (den Befehl direkt aufrufen)
                await self._cmd_status(update, context)
            
            elif callback_data == "refresh_status":
                # Status aktualisieren (die aktuelle Nachricht ersetzen)
                status = self.main_controller.get_status()
                
                # Status formatieren (gleicher Code wie in _cmd_status)
                bot_state = status.get('state', 'unbekannt')
                emoji = {
                    'running': '🟢', 'paused': '🟠', 'emergency': '🔴',
                    'error': '🔴', 'stopped': '⚪', 'ready': '🔵', 'initializing': '🔵'
                }.get(bot_state, '⚪')
                
                module_status = status.get('modules', {})
                
                message = (
                    f"<b>Trading Bot Status</b> {emoji}\n\n"
                    f"Status: <b>{bot_state.upper()}</b>\n"
                    f"Version: {status.get('version', 'unbekannt')}\n"
                    f"Laufzeit: {status.get('uptime', '00:00:00')}\n\n"
                    f"<b>Module:</b>\n"
                )
                
                for module_name, module_info in module_status.items():
                    module_state = module_info.get('status', 'unbekannt')
                    module_emoji = '🟢' if module_state == 'running' else '⚪'
                    message += f"{module_emoji} {module_name}: {module_state}\n"
                
                events = status.get('events', [])
                if events:
                    message += "\n<b>Letzte Ereignisse:</b>\n"
                    for event in events[-5:]:
                        event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                        event_type = event['type']
                        event_title = event['title']
                        message += f"• {event_time} [{event_type}] {event_title}\n"
                
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Aktualisieren", callback_data="refresh_status"),
                        InlineKeyboardButton("📊 Dashboard", callback_data="dashboard")
                    ],
                    [
                        InlineKeyboardButton("🟢 Start", callback_data="startbot"),
                        InlineKeyboardButton("🔴 Stop", callback_data="stopbot"),
                        InlineKeyboardButton("⏸️ Pause", callback_data="pausebot")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
            
            elif callback_data == "balance":
                # Kontostand anzeigen
                await self._cmd_balance(update, context)
            
            elif callback_data == "positions":
                # Positionen anzeigen
                await self._cmd_positions(update, context)
            
            elif callback_data == "performance":
                # Performance anzeigen
                await self._cmd_performance(update, context)
            
            elif callback_data == "help":
                # Hilfe anzeigen
                await self._cmd_help(update, context)
            
            elif callback_data == "dashboard":
                # Dashboard anzeigen
                await self._show_dashboard(query)
            
            elif callback_data == "refresh_positions":
                # Positionen aktualisieren
                await self._cmd_positions(update, context)
            
            elif callback_data == "close_all_positions":
                # Alle Positionen schließen
                await self._close_all_positions(query)
            
            else:
                # Unbekannter Callback
                self.logger.warning(f"Unbekannter Callback: {callback_data}")
                await query.message.reply_text(f"Unbekannte Aktion: {callback_data}")
        
        except Exception as e:
            error_message = f"Fehler bei der Verarbeitung des Callbacks: {str(e)}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            await query.message.reply_text(f"❌ Fehler: {error_message}")
    
    async def _show_dashboard(self, query):
        """Zeigt ein interaktives Dashboard an."""
        try:
            # Status vom MainController abrufen
            status = self.main_controller.get_status()
            bot_state = status.get('state', 'unbekannt')
            
            # Emoji basierend auf Status
            state_emojis = {
                'running': '🟢',
                'paused': '🟠',
                'emergency': '🔴',
                'error': '🔴',
                'stopped': '⚪',
                'ready': '🔵',
                'initializing': '🔵'
            }
            state_emoji = state_emojis.get(bot_state, '⚪')
            
            # Kontostand abrufen
            balance_result = await self._run_controller_method(self.main_controller._get_account_balance)
            balance_data = balance_result.get('balance', {}) if balance_result.get('status') == 'success' else {}
            
            # Positionen abrufen
            positions_result = await self._run_controller_method(self.main_controller._get_open_positions)
            positions = positions_result.get('positions', []) if positions_result.get('status') == 'success' else []
            
            # Performance-Metriken abrufen
            metrics_result = await self._run_controller_method(self.main_controller._get_performance_metrics)
            metrics = metrics_result.get('metrics', {}) if metrics_result.get('status') == 'success' else {}
            
            # Dashboard zusammenstellen
            message = (
                f"<b>📊 TRADING BOT DASHBOARD</b> {state_emoji}\n\n"
                f"<b>Status:</b> {bot_state.upper()}\n"
            )
            
            # Kontostand
            message += "\n<b>💰 Kontostand:</b>\n"
            if balance_data:
                for currency, amount in balance_data.items():
                    if isinstance(amount, (int, float)) and amount > 0:
                        if amount < 0.001:
                            formatted_amount = f"{amount:.8f}"
                        else:
                            formatted_amount = f"{amount:.2f}"
                        message += f"• {currency}: {formatted_amount}\n"
            else:
                message += "Keine Kontodaten verfügbar\n"
            
            # Offene Positionen
            open_pos_count = len(positions)
            message += f"\n<b>📈 Offene Positionen ({open_pos_count}):</b>\n"
            if positions:
                for i, pos in enumerate(positions[:3]):  # Max 3 anzeigen, um Platz zu sparen
                    symbol = pos.get('symbol', 'Unbekannt')
                    side = pos.get('side', 'Unbekannt')
                    side_emoji = '🔴' if side.lower() == 'short' else '🟢'
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_percent = pos.get('unrealized_pnl_percent', 0)
                    pnl_emoji = '📈' if pnl >= 0 else '📉'
                    
                    message += f"{side_emoji} {symbol} ({side}): {pnl_emoji} {pnl_percent:.2f}%\n"
                
                if open_pos_count > 3:
                    message += f"... und {open_pos_count - 3} weitere\n"
            else:
                message += "Keine offenen Positionen\n"
            
            # Performance-Zusammenfassung
            message += "\n<b>📊 Performance:</b>\n"
            if 'trading' in metrics:
                trading = metrics['trading']
                win_rate = trading.get('win_rate', 0) * 100
                total_trades = trading.get('total_trades', 0)
                total_pnl = trading.get('total_pnl', 0) * 100
                
                message += f"• Trades: {total_trades}\n"
                message += f"• Win-Rate: {win_rate:.1f}%\n"
                message += f"• Gesamt-PnL: {total_pnl:.2f}%\n"
            else:
                message += "Keine Performance-Daten verfügbar\n"
            
            # Letzte Ereignisse
            events = status.get('events', [])
            if events:
                message += "\n<b>🔔 Letzte Ereignisse:</b>\n"
                for event in events[-3:]:  # Zeige nur die letzten 3
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    event_type = event['type']
                    event_title = event['title']
                    
                    # Emoji je nach Event-Typ
                    type_emojis = {
                        'system': '🖥️',
                        'error': '❌',
                        'warning': '⚠️',
                        'trade': '💱',
                        'order': '📋',
                        'position': '📊',
                        'black_swan': '🦢',
                        'emergency': '🚨'
                    }
                    type_emoji = type_emojis.get(event_type, '📌')
                    
                    message += f"• {event_time} {type_emoji} {event_title}\n"
            
            # Dashboard-Aktionen
            keyboard = [
                [
                    InlineKeyboardButton("🟢 Start", callback_data="startbot"),
                    InlineKeyboardButton("🔴 Stop", callback_data="stopbot"),
                    InlineKeyboardButton("⏸️ Pause", callback_data="pausebot")
                ],
                [
                    InlineKeyboardButton("💰 Kontodetails", callback_data="balance"),
                    InlineKeyboardButton("📈 Alle Positionen", callback_data="positions")
                ],
                [
                    InlineKeyboardButton("📊 Performance", callback_data="performance"),
                    InlineKeyboardButton("🔄 Aktualisieren", callback_data="dashboard")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Zeitstempel hinzufügen
            message += f"\n<i>Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            
            # Dashboard anzeigen oder aktualisieren
            await query.edit_message_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        
        except Exception as e:
            error_message = f"Fehler beim Anzeigen des Dashboards: {str(e)}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            await query.message.reply_text(f"❌ Fehler: {error_message}")
    
    async def _close_all_positions(self, query):
        """Schließt alle offenen Positionen."""
        try:
            # Bestätigung anfordern
            message = (
                "⚠️ <b>Bestätigung erforderlich</b>\n\n"
                "Bist du sicher, dass du ALLE offenen Positionen schließen möchtest? "
                "Dies kann nicht rückgängig gemacht werden."
            )
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Ja, alle schließen", callback_data="confirm_close_all"),
                    InlineKeyboardButton("❌ Abbrechen", callback_data="cancel_close_all")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text(message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        
        except Exception as e:
            error_message = f"Fehler beim Schließen aller Positionen: {str(e)}"
            self.logger.error(error_message)
            await query.message.reply_text(f"❌ Fehler: {error_message}")
    
    async def _cmd_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt unbekannte Befehle."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        await update.message.reply_text(
            f"Unbekannter Befehl: {update.message.text}\n"
            f"Verwende /help, um eine Liste aller verfügbaren Befehle zu sehen."
        )
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt normale Textnachrichten."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await self._unauthorized_response(update)
            return
        
        # Hier könnten zukünftig Konversations-basierte Interaktionen implementiert werden
        await update.message.reply_text(
            "Ich verstehe nur Befehle. Verwende /help, um eine Liste aller verfügbaren Befehle zu sehen."
        )
    
    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt Fehler im Telegram-Bot."""
        self.logger.error(f"Fehler im Telegram-Bot: {context.error}")
        self.logger.error(traceback.format_exc())
        
        # Wenn möglich, Benutzer über den Fehler informieren
        if update and isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ Ein Fehler ist aufgetreten: {context.error}"
            )
    
    def _is_authorized(self, user_id: int) -> bool:
        """
        Prüft, ob ein Benutzer autorisiert ist.
        
        Args:
            user_id: Telegram-Benutzer-ID
        
        Returns:
            True, wenn der Benutzer autorisiert ist, sonst False
        """
        # Wenn keine erlaubten Benutzer konfiguriert sind, ist niemand autorisiert
        if not self.allowed_users:
            self.logger.warning(f"Zugriff verweigert für Benutzer {user_id}: Keine erlaubten Benutzer konfiguriert")
            return False
        
        is_authorized = user_id in self.allowed_users
        
        if not is_authorized:
            self.logger.warning(f"Zugriff verweigert für Benutzer {user_id}: Nicht autorisiert")
        
        return is_authorized
    
    async def _unauthorized_response(self, update: Update):
        """
        Sendet eine Antwort an nicht autorisierte Benutzer.
        
        Args:
            update: Das Update-Objekt
        """
        await update.effective_message.reply_text(
            "⛔ Du bist nicht autorisiert, diesen Bot zu verwenden."
        )
    
    async def _run_controller_method(self, method, *args, **kwargs):
        """
        Führt eine Methode des MainControllers im Hintergrund aus.
        
        Args:
            method: Die auszuführende Methode
            *args: Positionsargumente für die Methode
            **kwargs: Schlüsselwortargumente für die Methode
        
        Returns:
            Das Ergebnis der Methode
        """
        # Asynchron ausführen, um den Telegram-Thread nicht zu blockieren
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: method(*args, **kwargs))

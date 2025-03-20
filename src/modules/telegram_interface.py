# telegram_interface.py

import os
import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext, ContextTypes
from telegram.constants import ParseMode
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import io
from PIL import Image
import pandas as pd

# Für Headless-Server (ohne GUI)
matplotlib.use('Agg')

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
        
        # Verzeichnis für temporäre Grafiken
        self.charts_dir = Path('data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Trading-Informationen
        self.application.add_handler(CommandHandler("balance", self._balance_command))
        self.application.add_handler(CommandHandler("positions", self._positions_command))
        self.application.add_handler(CommandHandler("performance", self._performance_command))
        
        # Neue Befehle für Marktdaten und Berichte
        self.application.add_handler(CommandHandler("price", self._price_command))
        self.application.add_handler(CommandHandler("chart", self._chart_command))
        self.application.add_handler(CommandHandler("news", self._news_command))
        self.application.add_handler(CommandHandler("daily_report", self._daily_report_command))
        
        # Bot-Steuerungskommandos
        self.application.add_handler(CommandHandler("start_bot", self._start_bot_command))
        self.application.add_handler(CommandHandler("stop_bot", self._stop_bot_command))
        self.application.add_handler(CommandHandler("pause_bot", self._pause_bot_command))
        self.application.add_handler(CommandHandler("resume_bot", self._resume_bot_command))
        
        # Admin-Kommandos
        self.application.add_handler(CommandHandler("restart", self._restart_command))
        
        # Transkript-Verarbeitung
        self.application.add_handler(CommandHandler("process_transcript", self._process_transcript_command))
        
        # Handler für Inline-Tasten
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
        
        # Handler für normale Nachrichten (für Transkript-Aufzeichnung)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
        
        self.logger.info("Standard-Befehlshandler hinzugefügt")

    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für Inline-Tasten."""
        query = update.callback_query
        await query.answer()
        
        # Callback-Daten abrufen
        data = query.data
        
        if data.startswith("refresh_status"):
            # Status aktualisieren
            await self._status_command(update, context, is_callback=True)
        elif data.startswith("refresh_balance"):
            # Kontostand aktualisieren
            await self._balance_command(update, context, is_callback=True)
        elif data.startswith("refresh_positions"):
            # Positionen aktualisieren
            await self._positions_command(update, context, is_callback=True)
        elif data.startswith("start_bot"):
            # Bot starten
            await self._start_bot_command(update, context, is_callback=True)
        elif data.startswith("stop_bot"):
            # Bot stoppen
            await self._stop_bot_command(update, context, is_callback=True)
        elif data.startswith("chart_timeframe"):
            # Chart-Zeitrahmen ändern
            parts = data.split(":")
            if len(parts) >= 3:
                symbol = parts[1]
                timeframe = parts[2]
                await self._chart_command(update, context, symbol=symbol, timeframe=timeframe, is_callback=True)

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
            
            # Event-Loop für diesen Thread erstellen
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Polling starten
            loop.run_until_complete(self.application.run_polling(allowed_updates=Update.ALL_TYPES))
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

    def _broadcast_message(self, message: str, parse_mode: str = None):
        """
        Sendet eine Nachricht an alle autorisierten Benutzer.
        
        Args:
            message: Text der Nachricht
            parse_mode: Optional, Parse-Modus für die Nachricht (HTML, Markdown, etc.)
        """
        if not self.is_configured or not self.bot:
            self.logger.warning("Kann Broadcast nicht senden - Bot nicht konfiguriert")
            return
        
        for user_id in self.allowed_users:
            try:
                asyncio.run(self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=parse_mode
                ))
                self.logger.debug(f"Broadcast-Nachricht an Benutzer {user_id} gesendet")
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Broadcast-Nachricht an {user_id}: {str(e)}")

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
        
        # Erstelle Inline-Keyboard mit Schnellzugriffsbuttons
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="refresh_status"),
                InlineKeyboardButton("💰 Kontostand", callback_data="refresh_balance")
            ],
            [
                InlineKeyboardButton("📈 Positionen", callback_data="refresh_positions"),
                InlineKeyboardButton("❓ Hilfe", callback_data="show_help")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)

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
            "*Marktdaten & Berichte:*\n"
            "/price [Symbol] - Aktueller Preis (z.B. /price BTC)\n"
            "/chart [Symbol] [Zeitraum] - Zeigt ein Preisdiagramm\n"
            "/news [Thema] - Aktuelle Krypto-/Börsennachrichten\n"
            "/daily_report - Täglicher Zusammenfassungsbericht\n\n"
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

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """Handler für den /status Befehl."""
        if is_callback:
            query = update.callback_query
            chat_id = query.message.chat_id
            message_id = query.message.message_id
            
            if not self._check_authorized(update):
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="⛔ Du bist nicht autorisiert, diesen Bot zu verwenden."
                )
                return
        else:
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
                    message += f" {status_emoji} {module}: {module_status.get('status', 'Unbekannt')}\n"
                
                # Letzte Ereignisse
                events = status.get('events', [])
                if events:
                    message += "\n🔍 *Letzte Ereignisse*:\n"
                    for event in events[:5]:  # Nur die letzten 5 Ereignisse
                        event_time = datetime.fromisoformat(event.get('timestamp', '')).strftime('%H:%M:%S')
                        message += f" • {event_time} - {event.get('type', 'Unbekannt')}: {event.get('title', 'Kein Titel')}\n"
                
                # Inline-Keyboard für Aktionen
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Aktualisieren", callback_data="refresh_status"),
                        InlineKeyboardButton("📊 Positionen", callback_data="refresh_positions")
                    ],
                    [
                        InlineKeyboardButton("▶️ Starten", callback_data="start_bot"),
                        InlineKeyboardButton("⏹ Stoppen", callback_data="stop_bot")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message,
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await update.message.reply_text(
                        message,
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.MARKDOWN
                    )
            else:
                message = "⚠️ Kann Status nicht abrufen - MainController nicht verfügbar"
                
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        
        except Exception as e:
            self.logger.error(f"Fehler beim Status-Abruf: {str(e)}")
            error_message = f"❌ Fehler beim Abrufen des Status: {str(e)}"
            
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

    async def _balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """Handler für den /balance Befehl."""
        if is_callback:
            query = update.callback_query
            chat_id = query.message.chat_id
            message_id = query.message.message_id
            
            if not self._check_authorized(update):
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="⛔ Du bist nicht autorisiert, diesen Bot zu verwenden."
                )
                return
        else:
            if not self._check_authorized(update):
                await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
                return
        
        try:
            if self.main_controller and hasattr(self.main_controller, '_get_account_balance'):
                # Status-Nachricht senden
                if is_callback:
                    status_message = await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text="🔄 Rufe Kontostand ab..."
                    )
                else:
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
                                message += f" • {currency}: {amount}\n"
                    
                    if 'free' in balance:
                        message += "\n*Verfügbar:*\n"
                        for currency, amount in balance['free'].items():
                            if float(amount) > 0:
                                message += f" • {currency}: {amount}\n"
                    
                    if 'used' in balance:
                        message += "\n*In Verwendung:*\n"
                        for currency, amount in balance['used'].items():
                            if float(amount) > 0:
                                message += f" • {currency}: {amount}\n"
                    
                    # Balkendiagramm erstellen, wenn Daten verfügbar
                    if 'total' in balance and balance['total']:
                        try:
                            # Nur Assets mit Wert > 0 anzeigen
                            assets = [k for k, v in balance['total'].items() if float(v) > 0]
                            values = [float(balance['total'][k]) for k in assets]
                            
                            if assets and values:
                                # Erstelle Balkendiagramm
                                plt.figure(figsize=(10, 6))
                                bars = plt.bar(assets, values, color='skyblue')
                                plt.title('Assets im Portfolio')
                                plt.xlabel('Asset')
                                plt.ylabel('Betrag')
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                
                                # Füge Werte über den Balken hinzu
                                for bar in bars:
                                    height = bar.get_height()
                                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                             f'{height:.2f}', ha='center', va='bottom')
                                
                                # Speichere Diagramm in Puffer
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                plt.close()
                                
                                # Sende Nachricht mit Text
                                await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                                
                                # Sende Diagramm als separates Foto
                                await self.bot.send_photo(
                                    chat_id=(chat_id if is_callback else update.effective_chat.id),
                                    photo=buf
                                )
                                
                                return
                        except Exception as chart_error:
                            self.logger.error(f"Fehler beim Erstellen des Kontostand-Diagramms: {str(chart_error)}")
                    
                    # Wenn kein Diagramm erstellt wurde, nur die Nachricht senden
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(
                        f"❌ Fehler beim Abrufen des Kontostands: {balance_data.get('message', 'Unbekannter Fehler')}"
                    )
            else:
                message = "⚠️ Kann Kontostand nicht abrufen - MainController nicht verfügbar"
                
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        
        except Exception as e:
            self.logger.error(f"Fehler beim Kontostand-Abruf: {str(e)}")
            error_message = f"❌ Fehler beim Abrufen des Kontostands: {str(e)}"
            
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

    # Ergänzen Sie hier die restlichen Methoden der Klasse...
    # Die Methoden sind bereits in den Suchergebnissen vorhanden und sollten entsprechend eingefügt werden

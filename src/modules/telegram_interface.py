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

    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """Handler für den /positions Befehl."""
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
            if self.main_controller and hasattr(self.main_controller, '_get_open_positions'):
                # Status-Nachricht senden
                if is_callback:
                    status_message = await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text="🔄 Rufe offene Positionen ab..."
                    )
                else:
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
                    
                    # Sammle Daten für das Diagramm
                    symbols = []
                    pnls = []
                    colors = []
                    
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
                            color = 'green'
                        elif pnl_percent < 0:
                            emoji = "🔴"
                            color = 'red'
                        else:
                            emoji = "⚪"
                            color = 'gray'
                        
                        # Side formatieren
                        side_formatted = "LONG 📈" if side == 'long' else "SHORT 📉" if side == 'short' else side
                        
                        message += (
                            f"{emoji} *{symbol}* ({side_formatted})\n"
                            f" • Größe: {size} Kontrakte (Hebel: {leverage}x)\n"
                            f" • Einstieg: {entry_price}\n"
                            f" • Aktuell: {current_price}\n"
                            f" • PnL: {unrealized_pnl} ({pnl_percent:.2f}%)\n\n"
                        )
                        
                        # Daten für Diagramm hinzufügen
                        symbols.append(symbol)
                        pnls.append(float(unrealized_pnl))
                        colors.append(color)
                    
                    # PnL-Balkendiagramm erstellen
                    if symbols and pnls:
                        try:
                            plt.figure(figsize=(10, 6))
                            bars = plt.bar(symbols, pnls, color=colors)
                            plt.title('PnL offener Positionen')
                            plt.xlabel('Symbol')
                            plt.ylabel('Unrealisierter PnL')
                            plt.xticks(rotation=45)
                            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                            plt.tight_layout()
                            
                            # Füge Werte über den Balken hinzu
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width()/2., 
                                        height + 0.05 if height >= 0 else height - 0.5,
                                        f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
                            
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
                            self.logger.error(f"Fehler beim Erstellen des Positions-Diagramms: {str(chart_error)}")
                    
                    # Wenn kein Diagramm erstellt wurde, nur die Nachricht senden
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(
                        f"❌ Fehler beim Abrufen der Positionen: {positions_data.get('message', 'Unbekannter Fehler')}"
                    )
            else:
                message = "⚠️ Kann Positionen nicht abrufen - MainController nicht verfügbar"
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        except Exception as e:
            self.logger.error(f"Fehler beim Positionen-Abruf: {str(e)}")
            error_message = f"❌ Fehler beim Abrufen der Positionen: {str(e)}"
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

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
                        message += f" • Trades: {trading.get('total_trades', 0)}\n"
                        message += f" • Gewonnen: {trading.get('winning_trades', 0)}\n"
                        message += f" • Verloren: {trading.get('losing_trades', 0)}\n"
                        message += f" • Gewinnrate: {win_rate:.2f}%\n"
                        message += f" • Durchschn. Gewinn: {(trading.get('avg_win', 0) * 100):.2f}%\n"
                        message += f" • Durchschn. Verlust: {(trading.get('avg_loss', 0) * 100):.2f}%\n"
                        message += f" • Gesamt-PnL: {(trading.get('total_pnl', 0) * 100):.2f}%\n\n"
                        
                        # Performance-Diagramm erstellen
                        try:
                            # Erstelle Kreisdiagramm für Win/Loss-Verhältnis
                            win_loss_labels = ['Gewonnen', 'Verloren']
                            win_loss_sizes = [trading.get('winning_trades', 0), trading.get('losing_trades', 0)]
                            win_loss_colors = ['#4CAF50', '#F44336']
                            
                            plt.figure(figsize=(10, 6))
                            
                            # Subplot 1: Win/Loss Pie Chart
                            plt.subplot(1, 2, 1)
                            plt.pie(win_loss_sizes, labels=win_loss_labels, colors=win_loss_colors, autopct='%1.1f%%', startangle=90)
                            plt.axis('equal')
                            plt.title('Win/Loss Verhältnis')
                            
                            # Subplot 2: Durchschn. Gewinn/Verlust
                            plt.subplot(1, 2, 2)
                            avg_data = [trading.get('avg_win', 0) * 100, abs(trading.get('avg_loss', 0) * 100)]
                            plt.bar(['Durchschn. Gewinn', 'Durchschn. Verlust'], 
                                   avg_data, 
                                   color=['green', 'red'])
                            plt.title('Durchschn. Gewinn/Verlust (%)')
                            plt.ylabel('Prozent')
                            
                            # Layout anpassen
                            plt.tight_layout()
                            
                            # Speichere Diagramm in Puffer
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            plt.close()
                            
                            # Sende Nachricht mit Text
                            await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                            
                            # Sende Diagramm als separates Foto
                            await self.bot.send_photo(
                                chat_id=update.effective_chat.id,
                                photo=buf,
                                caption="Trading Performance Visualisierung"
                            )
                            
                            # Rest der Nachricht als separaten Text senden
                            rest_message = ""
                            
                            # Learning-Metriken
                            if 'learning' in metrics:
                                learning = metrics['learning']
                                rest_message += "🧠 *Learning Metrics*:\n"
                                for key, value in learning.items():
                                    rest_message += f" • {key}: {value}\n"
                                rest_message += "\n"
                            
                            # Steuer-Informationen
                            if 'tax' in metrics:
                                tax = metrics['tax']
                                rest_message += "💸 *Steuerinformationen*:\n"
                                if 'total_profit' in tax:
                                    rest_message += f" • Gesamtgewinn: {tax['total_profit']}\n"
                                if 'taxable_amount' in tax:
                                    rest_message += f" • Steuerpflichtiger Betrag: {tax['taxable_amount']}\n"
                                if 'tax_rate' in tax:
                                    rest_message += f" • Steuersatz: {tax['tax_rate']*100}%\n"
                                if 'estimated_tax' in tax:
                                    rest_message += f" • Geschätzte Steuer: {tax['estimated_tax']}\n"
                            
                            if rest_message:
                                await self.bot.send_message(
                                    chat_id=update.effective_chat.id,
                                    text=rest_message,
                                    parse_mode=ParseMode.MARKDOWN
                                )
                            
                            return
                        except Exception as chart_error:
                            self.logger.error(f"Fehler beim Erstellen des Performance-Diagramms: {str(chart_error)}")
                    
                    # Learning-Metriken
                    if 'learning' in metrics:
                        learning = metrics['learning']
                        message += "🧠 *Learning Metrics*:\n"
                        for key, value in learning.items():
                            message += f" • {key}: {value}\n"
                        message += "\n"
                    
                    # Steuer-Informationen
                    if 'tax' in metrics:
                        tax = metrics['tax']
                        message += "💸 *Steuerinformationen*:\n"
                        if 'total_profit' in tax:
                            message += f" • Gesamtgewinn: {tax['total_profit']}\n"
                        if 'taxable_amount' in tax:
                            message += f" • Steuerpflichtiger Betrag: {tax['taxable_amount']}\n"
                        if 'tax_rate' in tax:
                            message += f" • Steuersatz: {tax['tax_rate']*100}%\n"
                        if 'estimated_tax' in tax:
                            message += f" • Geschätzte Steuer: {tax['estimated_tax']}\n"
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(
                        f"❌ Fehler beim Abrufen der Performance-Metriken: {metrics_data.get('message', 'Unbekannter Fehler')}"
                    )
            else:
                await update.message.reply_text("⚠️ Kann Performance-Metriken nicht abrufen - MainController nicht verfügbar")
        except Exception as e:
            self.logger.error(f"Fehler beim Performance-Abruf: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen der Performance-Metriken: {str(e)}")

    async def _price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /price Befehl, um aktuelle Kryptokurse abzurufen."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            # Prüfen, ob ein Symbol angegeben wurde
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "ℹ️ Bitte gib ein Symbol an.\n"
                    "Beispiel: /price BTC oder /price ETH/USDT"
                )
                return
            
            symbol = context.args[0].upper()
            # Standardwährung hinzufügen, falls nicht angegeben
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            status_message = await update.message.reply_text(f"🔄 Rufe Preis für {symbol} ab...")
            
            if self.main_controller and hasattr(self.main_controller, 'data_pipeline'):
                # Aktuellen Preis abrufen
                data = self.main_controller.data_pipeline.get_crypto_data(symbol, '1m', 1)
                
                if data is not None and not data.empty:
                    last_price = data['close'].iloc[-1]
                    high_24h = data['high'].max()
                    low_24h = data['low'].min()
                    
                    # Preis-Änderung berechnen
                    if len(data) > 1:
                        price_change = (last_price / data['close'].iloc[0] - 1) * 100
                        change_text = f"{price_change:.2f}%"
                        change_emoji = "📈" if price_change >= 0 else "📉"
                    else:
                        change_text = "N/A"
                        change_emoji = "➖"
                    
                    message = (
                        f"💰 *{symbol} Kurs*\n\n"
                        f"{change_emoji} *Aktuell:* {last_price:.8f} USDT\n"
                        f"📊 *24h Hoch:* {high_24h:.8f} USDT\n"
                        f"📊 *24h Tief:* {low_24h:.8f} USDT\n"
                        f"📊 *Änderung:* {change_text}\n"
                        f"\nStand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
                    )
                    
                    # Inline-Keyboard für Chart-Optionen
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Chart anzeigen", callback_data=f"chart:{symbol}:1d")
                        ],
                        [
                            InlineKeyboardButton("1h", callback_data=f"chart_timeframe:{symbol}:1h"),
                            InlineKeyboardButton("4h", callback_data=f"chart_timeframe:{symbol}:4h"),
                            InlineKeyboardButton("1d", callback_data=f"chart_timeframe:{symbol}:1d"),
                            InlineKeyboardButton("1w", callback_data=f"chart_timeframe:{symbol}:1w")
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
                else:
                    await status_message.edit_text(f"❌ Keine Daten für {symbol} gefunden.")
            else:
                await update.message.reply_text("⚠️ Kann Preis nicht abrufen - DataPipeline nicht verfügbar")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Preises: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen des Preises: {str(e)}")

    async def _chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol=None, timeframe=None, is_callback=False):
        """Handler für den /chart Befehl, um Preisdiagramme anzuzeigen."""
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
            
            # Symbol und Zeitrahmen aus Befehlsargumenten extrahieren
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "ℹ️ Bitte gib ein Symbol und optional einen Zeitrahmen an.\n"
                    "Beispiel: /chart BTC 1d oder /chart ETH/USDT 4h"
                )
                return
            
            symbol = context.args[0].upper()
            # Standardwährung hinzufügen, falls nicht angegeben
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            timeframe = "1d"  # Standardzeitrahmen
            if len(context.args) > 1:
                timeframe = context.args[1].lower()
        
        try:
            # Status-Nachricht senden
            if is_callback:
                status_message = await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"🔄 Erstelle {timeframe}-Chart für {symbol}..."
                )
            else:
                status_message = await update.message.reply_text(f"🔄 Erstelle {timeframe}-Chart für {symbol}...")
            
            if self.main_controller and hasattr(self.main_controller, 'data_pipeline'):
                # Historische Daten abrufen
                # Anzahl der Datenpunkte je nach Zeitrahmen anpassen
                if timeframe == "1h": limit = 48
                elif timeframe == "4h": limit = 60
                elif timeframe == "1d": limit = 30
                elif timeframe == "1w": limit = 12
                else: limit = 30
                
                data = self.main_controller.data_pipeline.get_crypto_data(symbol, timeframe, limit)
                
                if data is not None and not data.empty:
                    # Preisdiagramm erstellen
                    plt.figure(figsize=(12, 8))
                    
                    # OHLC-Werte
                    dates = data.index
                    opens = data['open']
                    highs = data['high']
                    lows = data['low']
                    closes = data['close']
                    volumes = data['volume']
                    
                    # Berechne gleitende Durchschnitte
                    data['ma7'] = data['close'].rolling(window=7).mean()
                    data['ma21'] = data['close'].rolling(window=21).mean()
                    
                    # Preis-Chart (oberes Panel)
                    ax1 = plt.subplot(2, 1, 1)
                    ax1.plot(dates, closes, 'b-', label=f'{symbol} Preis')
                    ax1.plot(dates, data['ma7'], 'g-', label='7-Perioden MA')
                    ax1.plot(dates, data['ma21'], 'r-', label='21-Perioden MA')
                    
                    # Formatiere X-Achse
                    if len(dates) > 20:
                        ax1.set_xticks(dates[::len(dates)//10])
                    else:
                        ax1.set_xticks(dates)
                    
                    ax1.set_xticklabels([d.strftime('%d.%m') for d in dates[::max(1, len(dates)//10)]],
                                       rotation=45)
                    
                    # Einstellen des Layouts
                    ax1.set_title(f'{symbol} - {timeframe} Chart')
                    ax1.set_ylabel('Preis (USDT)')
                    ax1.legend(loc='upper left')
                    ax1.grid(True, alpha=0.3)
                    
                    # Volumen-Chart (unteres Panel)
                    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                    ax2.bar(dates, volumes, alpha=0.5, color='blue', label='Volumen')
                    ax2.set_ylabel('Volumen')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(bottom=0)
                    
                    plt.tight_layout()
                    
                    # Speichere Diagramm in Puffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plt.close()
                    
                    # Berechne Preisänderung
                    first_price = data['close'].iloc[0]
                    last_price = data['close'].iloc[-1]
                    price_change = (last_price / first_price - 1) * 100
                    
                    # Erstelle Tastatur für Zeitrahmenauswahl
                    keyboard = [
                        [
                            InlineKeyboardButton("1h", callback_data=f"chart_timeframe:{symbol}:1h"),
                            InlineKeyboardButton("4h", callback_data=f"chart_timeframe:{symbol}:4h"),
                            InlineKeyboardButton("1d", callback_data=f"chart_timeframe:{symbol}:1d"),
                            InlineKeyboardButton("1w", callback_data=f"chart_timeframe:{symbol}:1w")
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    caption = (
                        f"📊 {symbol} ({timeframe})\n"
                        f"Aktuell: {last_price:.8f}\n"
                        f"Änderung: {price_change:.2f}% {'📈' if price_change >= 0 else '📉'}\n"
                        f"Stand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
                    )
                    
                    # Sende Diagramm als Foto
                    if is_callback:
                        # Lösche alte Nachricht
                        await self.bot.delete_message(chat_id=chat_id, message_id=message_id)
                        
                        # Sende neue Nachricht mit Diagramm
                        await self.bot.send_photo(
                            chat_id=chat_id,
                            photo=buf,
                            caption=caption,
                            reply_markup=reply_markup
                        )
                    else:
                        # Lösche Status-Nachricht
                        await status_message.delete()
                        
                        # Sende neue Nachricht mit Diagramm
                        await self.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=buf,
                            caption=caption,
                            reply_markup=reply_markup
                        )
                else:
                    await status_message.edit_text(f"❌ Keine Daten für {symbol} ({timeframe}) gefunden.")
            else:
                message = "⚠️ Kann Chart nicht erstellen - DataPipeline nicht verfügbar"
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Charts: {str(e)}")
            error_message = f"❌ Fehler beim Erstellen des Charts: {str(e)}"
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

    async def _news_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /news Befehl, um aktuelle Krypto- und Börsennachrichten abzurufen."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            # Optionales Thema aus den Argumenten extrahieren
            topic = None
            if context.args and len(context.args) > 0:
                topic = context.args[0].lower()
            
            status_message = await update.message.reply_text("🔄 Rufe aktuelle Nachrichten ab...")
            
            if self.main_controller and hasattr(self.main_controller, 'data_pipeline') and hasattr(self.main_controller.data_pipeline, 'get_live_market_news'):
                # Nachrichten abrufen
                tickers = ["BTC", "ETH", "CRYPTO"]
                if topic:
                    tickers = [topic]
                
                news = self.main_controller.data_pipeline.get_live_market_news(tickers=tickers, limit=5)
                
                if news:
                    message = f"📰 *Aktuelle {topic or 'Krypto'}-Nachrichten*\n\n"
                    
                    for i, article in enumerate(news[:5], 1):
                        title = article.get('title', 'Kein Titel')
                        source = article.get('source', {}).get('name', 'Unbekannte Quelle')
                        url = article.get('url', '#')
                        date = datetime.fromisoformat(article.get('publishedAt', datetime.now().isoformat()).replace('Z', '+00:00'))
                        
                        message += f"{i}. *{title}*\n"
                        message += f"   Quelle: {source} | {date.strftime('%d.%m.%Y %H:%M')}\n\n"
                    
                    await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                else:
                    await status_message.edit_text(f"❌ Keine aktuellen Nachrichten gefunden.")
            else:
                await update.message.reply_text("⚠️ Kann Nachrichten nicht abrufen - News-Funktion nicht verfügbar")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Nachrichten: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Abrufen der Nachrichten: {str(e)}")

    async def _daily_report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /daily_report Befehl, um einen zusammenfassenden Tagesbericht anzuzeigen."""
        if not self._check_authorized(update):
            await update.message.reply_text("⛔ Du bist nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            status_message = await update.message.reply_text("🔄 Erstelle täglichen Bericht...")
            
            if self.main_controller:
                # Verschiedene Daten für den Bericht sammeln
                report_data = {}
                
                # 1. Performance-Daten
                if hasattr(self.main_controller, '_get_performance_metrics'):
                    metrics_data = self.main_controller._get_performance_metrics()
                    if metrics_data.get('status') == 'success':
                        report_data['metrics'] = metrics_data.get('metrics', {})
                
                # 2. Kontostand
                if hasattr(self.main_controller, '_get_account_balance'):
                    balance_data = self.main_controller._get_account_balance()
                    if balance_data.get('status') == 'success':
                        report_data['balance'] = balance_data.get('balance', {})
                
                # 3. Offene Positionen
                if hasattr(self.main_controller, '_get_open_positions'):
                    positions_data = self.main_controller._get_open_positions()
                    if positions_data.get('status') == 'success':
                        report_data['positions'] = positions_data.get('positions', [])
                
                # 4. Heutige Trades
                if hasattr(self.main_controller, '_get_today_trades'):
                    trades_data = self.main_controller._get_today_trades()
                    if trades_data.get('status') == 'success':
                        report_data['today_trades'] = trades_data.get('trades', [])
                
                # Bericht erstellen
                now = datetime.now()
                message = f"📋 *Täglicher Bericht - {now.strftime('%d.%m.%Y')}*\n\n"
                
                # Kontostand
                if 'balance' in report_data and report_data['balance'].get('total'):
                    message += "💰 *Kontostand:*\n"
                    
                    for currency, amount in report_data['balance']['total'].items():
                        if float(amount) > 0:
                            message += f" • {currency}: {amount}\n"
                    
                    message += "\n"
                
                # Performance
                if 'metrics' in report_data and 'trading' in report_data['metrics']:
                    trading = report_data['metrics']['trading']
                    daily_pnl = trading.get('daily_pnl', 0) * 100
                    win_rate = trading.get('win_rate', 0) * 100
                    
                    emoji = "📈" if daily_pnl >= 0 else "📉"
                    message += f"{emoji} *Tages-Performance:* {daily_pnl:.2f}%\n"
                    message += f"🎯 *Gewinnrate:* {win_rate:.2f}%\n\n"
                
                # Heutige Trades
                if 'today_trades' in report_data:
                    today_trades = report_data['today_trades']
                    message += f"🔄 *Heutige Trades:* {len(today_trades)}\n"
                    
                    if today_trades:
                        winning_trades = sum(1 for t in today_trades if t.get('pnl', 0) > 0)
                        losing_trades = sum(1 for t in today_trades if t.get('pnl', 0) < 0)
                        total_pnl = sum(t.get('pnl', 0) for t in today_trades)
                        
                        message += f" • Gewinner: {winning_trades}\n"
                        message += f" • Verlierer: {losing_trades}\n"
                        message += f" • Gesamt-PnL: {total_pnl:.2f}\n\n"
                        
                        # Trades-Diagramm erstellen
                        try:
                            # Sammle Daten für das Diagramm
                            trade_data = []
                            for trade in today_trades:
                                trade_data.append({
                                    'symbol': trade.get('symbol', 'Unbekannt'),
                                    'pnl': float(trade.get('pnl', 0)),
                                    'side': trade.get('side', 'Unbekannt'),
                                    'time': datetime.fromisoformat(trade.get('timestamp', ''))
                                })
                            
                            if trade_data:
                                # Sortiere nach Zeit
                                trade_data.sort(key=lambda x: x['time'])
                                
                                # Erstelle DataFrame
                                df = pd.DataFrame(trade_data)
                                
                                # Erstelle Gewinne/Verluste Balkendiagramm
                                plt.figure(figsize=(12, 10))
                                
                                # Subplot 1: PnL pro Trade
                                plt.subplot(2, 1, 1)
                                bars = plt.bar(range(len(df)), df['pnl'],
                                             color=['green' if pnl > 0 else 'red' for pnl in df['pnl']])
                                plt.title('PnL pro Trade')
                                plt.xlabel('Trade Nr.')
                                plt.ylabel('PnL')
                                plt.xticks(range(len(df)), [f"{i+1}" for i in range(len(df))])
                                
                                # Werte über den Balken
                                for i, bar in enumerate(bars):
                                    height = bar.get_height()
                                    plt.text(bar.get_x() + bar.get_width()/2., 
                                            0.05 if height < 0 else height + 0.05,
                                            f'{height:.2f}', ha='center', va='bottom')
                                
                                # Subplot 2: Kumulativer PnL
                                plt.subplot(2, 1, 2)
                                cumulative_pnl = df['pnl'].cumsum()
                                plt.plot(range(len(df)), cumulative_pnl, 'b-o', linewidth=2)
                                plt.title('Kumulativer PnL')
                                plt.xlabel('Trade Nr.')
                                plt.ylabel('Kumulativer PnL')
                                plt.xticks(range(len(df)), [f"{i+1}" for i in range(len(df))])
                                plt.grid(True, alpha=0.3)
                                
                                # Markiere Endwert
                                plt.annotate(f"{cumulative_pnl.iloc[-1]:.2f}", 
                                           xy=(len(df)-1, cumulative_pnl.iloc[-1]),
                                           xytext=(len(df)-1, cumulative_pnl.iloc[-1] + 0.5),
                                           arrowprops=dict(facecolor='black', shrink=0.05))
                                
                                plt.tight_layout()
                                
                                # Speichere Diagramm in Puffer
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                plt.close()
                                
                                # Sende Nachricht mit Text
                                await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
                                
                                # Sende Diagramm als separates Foto
                                await self.bot.send_photo(
                                    chat_id=update.effective_chat.id,
                                    photo=buf,
                                    caption=f"Trading-Performance am {now.strftime('%d.%m.%Y')}"
                                )
                                
                                # Verarbeite den Rest des Berichts
                                rest_message = ""
                                
                                # Offene Positionen
                                if 'positions' in report_data:
                                    positions = report_data['positions']
                                    rest_message += f"📊 *Offene Positionen:* {len(positions)}\n"
                                    
                                    if positions:
                                        for pos in positions[:3]:  # Top 3 Positionen
                                            symbol = pos.get('symbol', 'Unbekannt')
                                            side = pos.get('side', 'Unbekannt')
                                            side_emoji = "🟢" if side == 'long' else "🔴"
                                            unrealized_pnl = pos.get('unrealizedPnl', 0)
                                            
                                            rest_message += f" • {side_emoji} {symbol}: {unrealized_pnl}\n"
                                        
                                        if len(positions) > 3:
                                            rest_message += f" • ... und {len(positions) - 3} weitere\n"
                                        
                                        rest_message += "\n"
                                
                                # Endbemerkung
                                rest_message += "📱 Verwende /status oder /positions für mehr Details."
                                
                                if rest_message:
                                    await self.bot.send_message(
                                        chat_id=update.effective_chat.id,
                                        text=rest_message,
                                        parse_mode=ParseMode.MARKDOWN
                                    )
                                
                                return
                        except Exception as chart_error:
                            self.logger.error(f"Fehler beim Erstellen des Tagesberichts-Diagramms: {str(chart_error)}")
                
                # Offene Positionen
                if 'positions' in report_data:
                    positions = report_data['positions']
                    message += f"📊 *Offene Positionen:* {len(positions)}\n"
                    
                    if positions:
                        for pos in positions[:3]:  # Top 3 Positionen
                            symbol = pos.get('symbol', 'Unbekannt')
                            side = pos.get('side', 'Unbekannt')
                            side_emoji = "🟢" if side == 'long' else "🔴"
                            unrealized_pnl = pos.get('unrealizedPnl', 0)
                            
                            message += f" • {side_emoji} {symbol}: {unrealized_pnl}\n"
                        
                        if len(positions) > 3:
                            message += f" • ... und {len(positions) - 3} weitere\n"
                        
                        message += "\n"
                
                # Endbemerkung
                message += "📱 Verwende /status oder /positions für mehr Details."
                
                await status_message.edit_text(message, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text("⚠️ Kann Bericht nicht erstellen - MainController nicht verfügbar")
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Tagesberichts: {str(e)}")
            await update.message.reply_text(f"❌ Fehler beim Erstellen des Tagesberichts: {str(e)}")

    async def _start_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """Handler für den /start_bot Befehl."""
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
            if self.main_controller and hasattr(self.main_controller, 'start'):
                # Status-Nachricht senden
                if is_callback:
                    status_message = await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text="🔄 Starte Trading Bot..."
                    )
                else:
                    status_message = await update.message.reply_text("🔄 Starte Trading Bot...")
                
                # Auto-Trading-Parameter aus Nachricht extrahieren
                auto_trade = True
                if not is_callback and context.args and len(context.args) > 0:
                    arg = context.args[0].lower()
                    if arg in ["false", "no", "0", "off"]:
                        auto_trade = False
                
                # Bot starten
                success = self.main_controller.start(auto_trade=auto_trade)
                
                if success:
                    message = (
                        f"✅ Trading Bot erfolgreich gestartet!\n"
                        f"Auto-Trading: {'Aktiviert' if auto_trade else 'Deaktiviert'}"
                    )
                    
                    if is_callback:
                        await status_message.edit_text(message)
                    else:
                        await status_message.edit_text(message)
                else:
                    message = "❌ Fehler beim Starten des Trading Bots"
                    if is_callback:
                        await status_message.edit_text(message)
                    else:
                        await status_message.edit_text(message)
            else:
                message = "⚠️ Kann Bot nicht starten - MainController nicht verfügbar"
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Start: {str(e)}")
            error_message = f"❌ Fehler beim Starten des Bots: {str(e)}"
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

    async def _stop_bot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_callback=False):
        """Handler für den /stop_bot Befehl."""
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
            if self.main_controller and hasattr(self.main_controller, 'stop'):
                # Status-Nachricht senden
                if is_callback:
                    status_message = await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text="🔄 Stoppe Trading Bot..."
                    )
                else:
                    status_message = await update.message.reply_text("🔄 Stoppe Trading Bot...")
                
                # Bot stoppen
                success = self.main_controller.stop()
                
                if success:
                    message = "✅ Trading Bot erfolgreich gestoppt!"
                    if is_callback:
                        await status_message.edit_text(message)
                    else:
                        await status_message.edit_text(message)
                else:
                    message = "❌ Fehler beim Stoppen des Trading Bots"
                    if is_callback:
                        await status_message.edit_text(message)
                    else:
                        await status_message.edit_text(message)
            else:
                message = "⚠️ Kann Bot nicht stoppen - MainController nicht verfügbar"
                if is_callback:
                    await self.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=message
                    )
                else:
                    await update.message.reply_text(message)
        except Exception as e:
            self.logger.error(f"Fehler beim Bot-Stopp: {str(e)}")
            error_message = f"❌ Fehler beim Stoppen des Bots: {str(e)}"
            if is_callback:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=error_message
                )
            else:
                await update.message.reply_text(error_message)

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

    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für normale Nachrichten (für Transkript-Aufzeichnung)."""
        if not self._check_authorized(update):
            return
        
        # Hier könnte eine Implementierung zur Transkript-Aufzeichnung erfolgen
        # Beispiel: Speichern der Nachricht in einer Transkriptdatei

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
            
            # Extrahiere Datum aus Dateinamen
            match = re.search(r'transcript_(\d{8}).*\.txt', file_path.name)
            if match:
                date_str = match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
                
                # Prüfe Dateigröße
                size = file_path.stat().st_size
                
                transcripts.append({
                    'path': str(file_path),
                    'date': date,
                    'size': size
                })
        
        return transcripts
    except Exception as e:
        self.logger.error(f"Fehler beim Abrufen der Transkripte: {str(e)}")
        return []

# Ende der Klasse TelegramInterface

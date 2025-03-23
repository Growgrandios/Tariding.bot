# telegram_interface.py

import os
import sys
import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Union, Callable
import traceback
from datetime import datetime

from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)

class TelegramInterface:
    """
    Telegram-Schnittstelle für den Trading Bot.
    Ermöglicht die Steuerung und Überwachung über Telegram.
    """
    
    def __init__(self, config: Dict[str, Any], controller=None):
        """
        Initialisiert die Telegram-Schnittstelle.
        
        Args:
            config: Konfigurationseinstellungen mit Bot-Token und erlaubten Benutzern
            controller: Referenz zum MainController
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface...")
        
        # Konfiguration
        self.config = config or {}
        self.controller = controller
        
        # Telegram Bot Token
        self.token = self.config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        if not self.token:
            self.logger.error("Kein Telegram Bot Token gefunden!")
            self.is_ready = False
            return
            
        # Autorisierte Benutzer
        self.allowed_users = self.config.get('allowed_users', [])
        self.admin_users = self.config.get('admin_users', [])
        
        # Status
        self.is_ready = False
        self.is_running = False
        
        # Application und Bot
        try:
            # In v20+ we use Application instead of Updater
            self.application = Application.builder().token(self.token).build()
            self.bot = self.application.bot
            
            # Command Handlers registrieren
            self._register_handlers()
            
            self.is_ready = True
            self.logger.info("TelegramInterface erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_ready = False
        
        # Registrierte Befehle
        self.commands = {}
    
    def _register_handlers(self):
        """Registriert alle Command- und Callback-Handler."""
        # Basis-Befehle
        self.application.add_handler(CommandHandler("start", self._cmd_welcome))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        
        # Bot-Steuerungsbefehle
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("startbot", self._cmd_start_bot))
        self.application.add_handler(CommandHandler("stopbot", self._cmd_stop_bot))
        self.application.add_handler(CommandHandler("pausebot", self._cmd_pause_bot))
        self.application.add_handler(CommandHandler("resumebot", self._cmd_resume_bot))
        
        # Trading-Befehle
        self.application.add_handler(CommandHandler("balance", self._cmd_balance))
        self.application.add_handler(CommandHandler("positions", self._cmd_positions))
        self.application.add_handler(CommandHandler("orders", self._cmd_orders))
        
        # Analyse-Befehle
        self.application.add_handler(CommandHandler("performance", self._cmd_performance))
        self.application.add_handler(CommandHandler("market", self._cmd_market))
        self.application.add_handler(CommandHandler("prediction", self._cmd_prediction))
        
        # Callback-Handler für Buttons
        self.application.add_handler(CallbackQueryHandler(self._handle_button_press))
        
        # Fehlerhandler
        self.application.add_error_handler(self._error_handler)
    
    def start(self):
        """Startet den Telegram Bot im Hintergrund."""
        if not self.is_ready:
            self.logger.error("TelegramInterface ist nicht bereit zum Start")
            return False
            
        if self.is_running:
            self.logger.warning("TelegramInterface läuft bereits")
            return True
            
        try:
            # Bot im Hintergrund starten
            self.application.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)
            self.is_running = True
            self.logger.info("TelegramInterface erfolgreich gestartet")
            
            # Admin-Benutzer benachrichtigen
            for admin_id in self.admin_users:
                try:
                    self.send_message(
                        admin_id,
                        "🤖 Trading Bot wurde gestartet und ist jetzt über Telegram steuerbar.",
                        disable_notification=True
                    )
                except Exception as e:
                    self.logger.warning(f"Konnte Admin {admin_id} nicht benachrichtigen: {str(e)}")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def stop(self):
        """Stoppt den Telegram Bot."""
        if not self.is_running:
            self.logger.warning("TelegramInterface läuft nicht")
            return True
            
        try:
            # In v20+ we use stop() on the application
            self.application.stop()
            self.is_running = False
            self.logger.info("TelegramInterface erfolgreich gestoppt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Telegram Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    # Hilfsfunktionen
    
    def _is_user_authorized(self, user_id: int) -> bool:
        """Prüft, ob ein Benutzer autorisiert ist."""
        return str(user_id) in self.allowed_users or str(user_id) in self.admin_users
    
    def _is_admin(self, user_id: int) -> bool:
        """Prüft, ob ein Benutzer Admin-Rechte hat."""
        return str(user_id) in self.admin_users
    
    async def send_message(self, chat_id: int, text: str, parse_mode: str = ParseMode.HTML,
                     reply_markup: Any = None, disable_notification: bool = False) -> Optional[Any]:
        """Sendet eine Nachricht an einen Chat."""
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_notification=disable_notification
            )
        except Exception as e:
            self.logger.error(f"Fehler beim Senden einer Nachricht an {chat_id}: {str(e)}")
            return None
    
    def register_commands(self, commands: Dict[str, Callable]):
        """Registriert Befehle vom MainController."""
        self.commands.update(commands)
        self.logger.info(f"Befehle registriert: {', '.join(commands.keys())}")
    
    # Button-Handler
    
    async def _handle_button_press(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet Klicks auf Inline-Buttons."""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Autorisierung prüfen
        if not self._is_user_authorized(user_id):
            await query.answer("❌ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            # Callback-Daten extrahieren
            callback_data = query.data
            
            # Bestätigung anzeigen
            await query.answer()
            
            # Callback-Typen verarbeiten
            if callback_data == "menu":
                # Hauptmenü anzeigen
                await self._show_main_menu(query.message)
            elif callback_data == "status":
                # Status abrufen (verzögerte Antwort über Edit)
                await query.edit_message_text("⏳ Aktualisiere Status...")
                await self._cmd_status(update, context)
            elif callback_data == "balance":
                # Kontostand abrufen
                await query.edit_message_text("⏳ Rufe Kontostand ab...")
                await self._cmd_balance(update, context)
            elif callback_data == "positions":
                # Positionen abrufen
                await query.edit_message_text("⏳ Rufe Positionen ab...")
                await self._cmd_positions(update, context)
            elif callback_data == "orders":
                # Orders abrufen
                await query.edit_message_text("⏳ Rufe Orders ab...")
                await self._cmd_orders(update, context)
            elif callback_data == "performance":
                # Performance abrufen
                await query.edit_message_text("⏳ Rufe Performance-Daten ab...")
                await self._cmd_performance(update, context)
            elif callback_data == "startbot":
                # Bot starten
                await query.edit_message_text("⏳ Starte Bot...")
                await self._cmd_start_bot(update, context)
            elif callback_data == "stopbot":
                # Bot stoppen
                await query.edit_message_text("⏳ Stoppe Bot...")
                await self._cmd_stop_bot(update, context)
            elif callback_data == "pausebot":
                # Bot pausieren
                await query.edit_message_text("⏳ Pausiere Bot...")
                await self._cmd_pause_bot(update, context)
            elif callback_data == "resumebot":
                # Bot fortsetzen
                await query.edit_message_text("⏳ Setze Bot fort...")
                await self._cmd_resume_bot(update, context)
            elif callback_data == "market":
                # Marktdaten abrufen
                await query.edit_message_text("⏳ Rufe Marktdaten ab...")
                await self._cmd_market(update, context)
            elif callback_data == "prediction":
                # Prognose abrufen
                await query.edit_message_text("⏳ Rufe Prognose ab...")
                await self._cmd_prediction(update, context)
            else:
                # Unbekannter Callback
                await query.edit_message_text(f"Unbekannte Aktion: {callback_data}")
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Button-Klicks: {str(e)}")
            try:
                await query.edit_message_text(f"❌ Fehler: {str(e)}")
            except:
                pass
    
    # Command-Handler
    
    async def _cmd_welcome(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Begrüßt den Benutzer und zeigt Hauptmenü an."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            await update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        user_name = update.effective_user.first_name
        
        welcome_text = (
            f"👋 Willkommen, {user_name}!\n\n"
            "🤖 <b>Trading Bot</b>\n\n"
            "Ich bin Ihr persönlicher Trading-Assistent. Über mich können Sie den Trading Bot steuern und überwachen.\n\n"
            "Tippen Sie /help für eine Liste aller verfügbaren Befehle oder nutzen Sie das Menü unten."
        )
        
        # Menü-Buttons erstellen
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💰 Kontostand", callback_data="balance")
            ],
            [
                InlineKeyboardButton("🚀 Bot starten", callback_data="startbot"),
                InlineKeyboardButton("🛑 Bot stoppen", callback_data="stopbot")
            ],
            [
                InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                InlineKeyboardButton("📋 Orders", callback_data="orders")
            ],
            [
                InlineKeyboardButton("📉 Performance", callback_data="performance"),
                InlineKeyboardButton("🌍 Markt", callback_data="market")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_html(
            welcome_text,
            reply_markup=reply_markup
        )
        
        update.message.reply_html(
            welcome_text,
            reply_markup=reply_markup
        )
    
    def _cmd_help(self, update: Update, context: CallbackContext):
        """Zeigt Hilfetext mit allen verfügbaren Befehlen."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        help_text = (
            "🤖 <b>Trading Bot - Verfügbare Befehle</b>\n\n"
            "<b>Allgemeine Befehle:</b>\n"
            "/start - Startmenü anzeigen\n"
            "/help - Diese Hilfe anzeigen\n"
            "/status - Aktuellen Status des Bots anzeigen\n\n"
            
            "<b>Bot-Steuerung:</b>\n"
            "/startbot - Trading Bot starten\n"
            "/stopbot - Trading Bot stoppen\n"
            "/pausebot - Trading pausieren (Überwachung bleibt aktiv)\n"
            "/resumebot - Trading fortsetzen\n\n"
            
            "<b>Trading-Informationen:</b>\n"
            "/balance - Kontostand anzeigen\n"
            "/positions - Offene Positionen anzeigen\n"
            "/orders - Offene Orders anzeigen\n\n"
            
            "<b>Analyse & Prognose:</b>\n"
            "/performance - Performance-Metriken anzeigen\n"
            "/market - Marktanalyse anzeigen\n"
            "/prediction - Aktuelle Prognosen anzeigen\n"
        )
        
        # Admin-Befehle für Administratoren
        if self._is_admin(user_id):
            help_text += (
                "\n<b>Admin-Befehle:</b>\n"
                "/restart - Bot neu starten\n"
                "/config - Konfiguration anzeigen/ändern\n"
                "/logs - Aktuelle Logs anzeigen\n"
            )
        
        keyboard = [[InlineKeyboardButton("🔙 Zurück zum Menü", callback_data="menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_html(
            help_text,
            reply_markup=reply_markup
        )
    
    def _cmd_status(self, update: Update, context: CallbackContext):
        """Zeigt den aktuellen Status des Bots an."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe Statusdaten ab...")
        
        try:
            # Status vom Controller abrufen
            status = self.controller.get_status()
            
            # Status-Emoji basierend auf Bot-Status
            status_emoji = {
                "initializing": "🔄",
                "ready": "✅",
                "running": "▶️",
                "paused": "⏸️",
                "stopping": "⏹️",
                "error": "❌",
                "maintenance": "🔧",
                "emergency": "🚨"
            }.get(status.get('state', 'unknown'), "❓")
            
            # Module-Status
            modules_status = "\n".join([
                f"- {name}: {module_status.get('status', 'unknown')}"
                for name, module_status in status.get('modules', {}).items()
            ])
            
            # Trading-Status
            trading_status = status.get('trading', {})
            trading_active = trading_status.get('active', False)
            trading_mode = trading_status.get('mode', 'unknown')
            
            # Antwort-Text erstellen
            status_text = (
                f"{status_emoji} <b>Bot-Status: {status.get('state', 'unknown').upper()}</b>\n\n"
                f"<b>Trading:</b> {'Aktiv' if trading_active else 'Inaktiv'}\n"
                f"<b>Modus:</b> {trading_mode}\n"
                f"<b>Laufzeit:</b> {status.get('uptime', '0:00:00')}\n"
                f"<b>Letzte Aktivität:</b> {status.get('last_activity', 'Keine')}\n\n"
                f"<b>Module:</b>\n{modules_status}\n\n"
                f"<b>Ereignisse:</b>\n"
                f"- Trades heute: {status.get('trades_today', 0)}\n"
                f"- Fehler heute: {status.get('errors_today', 0)}\n"
                f"- Warnungen: {status.get('warnings', 0)}\n"
            )
            
            # Buttons für weitere Aktionen
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Aktualisieren", callback_data="status"),
                    InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                ],
                [
                    InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                    InlineKeyboardButton("💰 Kontostand", callback_data="balance")
                ]
            ]
            
            # Trading-Steuerungsbuttons basierend auf aktuellem Status
            if status.get('state') == "running":
                keyboard.append([
                    InlineKeyboardButton("⏸️ Pausieren", callback_data="pausebot"),
                    InlineKeyboardButton("⏹️ Stoppen", callback_data="stopbot")
                ])
            elif status.get('state') == "paused":
                keyboard.append([
                    InlineKeyboardButton("▶️ Fortsetzen", callback_data="resumebot"),
                    InlineKeyboardButton("⏹️ Stoppen", callback_data="stopbot")
                ])
            elif status.get('state') in ["ready", "error"]:
                keyboard.append([
                    InlineKeyboardButton("▶️ Starten", callback_data="startbot")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Nachricht aktualisieren
            message.edit_text(
                status_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Status: {str(e)}")
            message.edit_text(f"❌ Fehler beim Abrufen des Status: {str(e)}")
    
    def _cmd_start_bot(self, update: Update, context: CallbackContext):
        """Startet den Trading Bot."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Starte Trading Bot...")
        
        try:
            # Befehl vom Controller ausführen
            if 'start' in self.commands:
                result = self.commands['start']()
                
                if result:
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Status", callback_data="status"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("⏸️ Pausieren", callback_data="pausebot"),
                            InlineKeyboardButton("⏹️ Stoppen", callback_data="stopbot")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        "✅ Trading Bot erfolgreich gestartet!",
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Fehler beim Starten des Trading Bots.")
            else:
                message.edit_text("❌ Start-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Bots: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_stop_bot(self, update: Update, context: CallbackContext):
        """Stoppt den Trading Bot."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Stoppe Trading Bot...")
        
        try:
            # Befehl vom Controller ausführen
            if 'stop' in self.commands:
                result = self.commands['stop']()
                
                if result:
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Status", callback_data="status"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("▶️ Starten", callback_data="startbot")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        "✅ Trading Bot erfolgreich gestoppt!",
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Fehler beim Stoppen des Trading Bots.")
            else:
                message.edit_text("❌ Stop-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Bots: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_pause_bot(self, update: Update, context: CallbackContext):
        """Pausiert den Trading Bot."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Pausiere Trading Bot...")
        
        try:
            # Befehl vom Controller ausführen
            if 'pause' in self.commands:
                result = self.commands['pause']()
                
                if result:
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Status", callback_data="status"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("▶️ Fortsetzen", callback_data="resumebot"),
                            InlineKeyboardButton("⏹️ Stoppen", callback_data="stopbot")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        "⏸️ Trading Bot pausiert. Überwachung bleibt aktiv.",
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Fehler beim Pausieren des Trading Bots.")
            else:
                message.edit_text("❌ Pause-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Pausieren des Bots: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_resume_bot(self, update: Update, context: CallbackContext):
        """Setzt den pausierten Trading Bot fort."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Setze Trading Bot fort...")
        
        try:
            # Befehl vom Controller ausführen
            if 'resume' in self.commands:
                result = self.commands['resume']()
                
                if result:
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Status", callback_data="status"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("⏸️ Pausieren", callback_data="pausebot"),
                            InlineKeyboardButton("⏹️ Stoppen", callback_data="stopbot")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        "▶️ Trading Bot fortgesetzt!",
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Fehler beim Fortsetzen des Trading Bots.")
            else:
                message.edit_text("❌ Resume-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Fortsetzen des Bots: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_balance(self, update: Update, context: CallbackContext):
        """Zeigt den aktuellen Kontostand."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe Kontostand ab...")
        
        try:
            # Führe den balance-Befehl aus, falls registriert
            if 'balance' in self.commands:
                balance = self.commands['balance']()
                
                if balance:
                    # Balancedaten formatieren
                    account_type = balance.get('account_type', 'Unbekannt')
                    total = balance.get('total', {})
                    free = balance.get('free', {})
                    used = balance.get('used', {})
                    
                    # USDT-Bilanz hervorheben
                    usdt_total = total.get('USDT', 0)
                    usdt_free = free.get('USDT', 0)
                    usdt_used = used.get('USDT', 0)
                    
                    # Andere Assets
                    other_assets = []
                    for currency, amount in total.items():
                        if currency != 'USDT' and float(amount) > 0:
                            free_amount = free.get(currency, 0)
                            used_amount = used.get(currency, 0)
                            other_assets.append(f"- {currency}: {amount:.6f} (Verfügbar: {free_amount:.6f}, In Verwendung: {used_amount:.6f})")
                    
                    other_assets_text = "\n".join(other_assets) if other_assets else "Keine"
                    
                    # Antwort-Text erstellen
                    balance_text = (
                        f"💰 <b>Kontostand ({account_type})</b>\n\n"
                        f"<b>USDT Gesamt:</b> {usdt_total:.2f}\n"
                        f"<b>USDT Verfügbar:</b> {usdt_free:.2f}\n"
                        f"<b>USDT In Verwendung:</b> {usdt_used:.2f}\n\n"
                        
                        f"<b>Andere Assets:</b>\n"
                        f"{other_assets_text}\n\n"
                        
                        f"<i>Stand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</i>"
                    )
                    
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("🔄 Aktualisieren", callback_data="balance"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                            InlineKeyboardButton("📋 Orders", callback_data="orders")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        balance_text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Konnte Kontostand nicht abrufen.")
            else:
                message.edit_text("❌ Balance-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_positions(self, update: Update, context: CallbackContext):
        """Zeigt offene Positionen."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe offene Positionen ab...")
        
        try:
            # Führe den positions-Befehl aus, falls registriert
            if 'positions' in self.commands:
                positions = self.commands['positions']()
                
                if positions:
                    # Positionen formatieren
                    if len(positions) > 0:
                        positions_text = "<b>📈 Offene Positionen</b>\n\n"
                        
                        for pos in positions:
                            symbol = pos.get('symbol', 'Unbekannt')
                            side = pos.get('side', 'Unbekannt')
                            size = pos.get('contracts', 0)
                            entry_price = pos.get('entryPrice', 0)
                            current_price = pos.get('markPrice', 0)
                            pnl = pos.get('unrealizedPnl', 0)
                            
                            # Emojis basierend auf Position und PnL
                            side_emoji = "🔴" if side.lower() == "short" else "🟢"
                            pnl_emoji = "📈" if float(pnl) > 0 else "📉"
                            
                            positions_text += (
                                f"{side_emoji} <b>{symbol}</b> ({side.upper()})\n"
                                f"Größe: {size} Kontrakte\n"
                                f"Einstieg: {float(entry_price):.2f}\n"
                                f"Aktuell: {float(current_price):.2f}\n"
                                f"{pnl_emoji} PnL: {float(pnl):.2f} USDT\n\n"
                            )
                    else:
                        positions_text = "📊 <b>Keine offenen Positionen</b>"
                    
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("🔄 Aktualisieren", callback_data="positions"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("💰 Kontostand", callback_data="balance"),
                            InlineKeyboardButton("📋 Orders", callback_data="orders")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        positions_text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Konnte offene Positionen nicht abrufen.")
            else:
                message.edit_text("❌ Positions-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Positionen: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_orders(self, update: Update, context: CallbackContext):
        """Zeigt offene Orders."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe offene Orders ab...")
        
        try:
            # Führe den orders-Befehl aus, falls registriert
            if 'orders' in self.commands:
                orders = self.commands['orders']()
                
                if orders:
                    # Orders formatieren
                    if len(orders) > 0:
                        orders_text = "<b>📋 Offene Orders</b>\n\n"
                        
                        for order in orders:
                            symbol = order.get('symbol', 'Unbekannt')
                            order_id = order.get('id', 'Unbekannt')
                            order_type = order.get('type', 'Unbekannt')
                            side = order.get('side', 'Unbekannt')
                            price = order.get('price', 0)
                            amount = order.get('amount', 0)
                            
                            # Emojis basierend auf Order-Typ und Seite
                            type_emoji = "📍" if order_type.lower() == "limit" else "⚡"
                            side_emoji = "🔴" if side.lower() == "sell" else "🟢"
                            
                            orders_text += (
                                f"{type_emoji}{side_emoji} <b>{symbol}</b>\n"
                                f"ID: {order_id[:8]}...\n"
                                f"Typ: {order_type.upper()} {side.upper()}\n"
                                f"Preis: {float(price):.2f}\n"
                                f"Menge: {float(amount):.6f}\n\n"
                            )
                    else:
                        orders_text = "📋 <b>Keine offenen Orders</b>"
                    
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("🔄 Aktualisieren", callback_data="orders"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("💰 Kontostand", callback_data="balance"),
                            InlineKeyboardButton("📈 Positionen", callback_data="positions")
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        orders_text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Konnte offene Orders nicht abrufen.")
            else:
                message.edit_text("❌ Orders-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Orders: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_performance(self, update: Update, context: CallbackContext):
        """Zeigt Performance-Metriken."""
        # Implementierung hier...
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        update.message.reply_text("Diese Funktion wird in Kürze implementiert.")
    
    def _cmd_market(self, update: Update, context: CallbackContext):
        """Zeigt Marktanalyse-Daten."""
        # Implementierung hier...
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        update.message.reply_text("Diese Funktion wird in Kürze implementiert.")
    
    def _cmd_prediction(self, update: Update, context: CallbackContext):
        """Zeigt aktuelle Prognosen."""
        # Implementierung hier...
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        update.message.reply_text("Diese Funktion wird in Kürze implementiert.")
    
    def _show_main_menu(self, message):
        """Zeigt das Hauptmenü mit allen verfügbaren Optionen."""
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💰 Kontostand", callback_data="balance")
            ],
            [
                InlineKeyboardButton("🚀 Bot starten", callback_data="startbot"),
                InlineKeyboardButton("🛑 Bot stoppen", callback_data="stopbot")
            ],
            [
                InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                InlineKeyboardButton("📋 Orders", callback_data="orders")
            ],
            [
                InlineKeyboardButton("📉 Performance", callback_data="performance"),
                InlineKeyboardButton("🌍 Markt", callback_data="market")
            ],
            [
                InlineKeyboardButton("🔮 Prognose", callback_data="prediction")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message.edit_text(
            "🤖 <b>Trading Bot - Hauptmenü</b>\n\n"
            "Wählen Sie eine Option:",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sendet eine Benachrichtigung an alle autorisierten Benutzer.
        
        Args:
            title: Titel der Benachrichtigung
            message: Nachrichtentext
            priority: Priorität ('low', 'normal', 'high', 'critical')
        """
        if not self.is_running:
            self.logger.warning("TelegramInterface ist nicht aktiv. Benachrichtigung wird nicht gesendet.")
            return
        
        # Prioritäts-Emojis
        priority_icons = {
            'low': 'ℹ️',
            'normal': '📢',
            'high': '⚠️',
            'critical': '🚨'
        }
        
        icon = priority_icons.get(priority, '📢')
        
        notification_text = f"{icon} <b>{title}</b>\n\n{message}"
        
        # An alle autorisierten Benutzer senden
        for user_id in self.allowed_users:
            try:
                # Hohe Priorität = mit Benachrichtigung, sonst ohne
                disable_notification = priority.lower() in ['low', 'normal']
                self.send_message(
                    user_id,
                    notification_text,
                    parse_mode=ParseMode.HTML,
                    disable_notification=disable_notification
                )
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Benachrichtigung an {user_id}: {str(e)}")
    
    def _error_handler(self, update: object, context: CallbackContext):
        """Behandelt Fehler in der Telegram-Bot-Verarbeitung."""
        try:
            if update:
                # Wenn der Fehler innerhalb eines Updates aufgetreten ist
                if isinstance(update, Update) and update.effective_message:
                    chat_id = update.effective_message.chat_id
                    update.effective_message.reply_text(
                        "❌ Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später noch einmal."
                    )
                # Log the error
                self.logger.error(f"Update {update} caused error: {context.error}")
            else:
                self.logger.error(f"Error without update: {context.error}")
            
            # Traceback ins Log
            self.logger.error(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"Fehler im Error-Handler selbst: {str(e)}")

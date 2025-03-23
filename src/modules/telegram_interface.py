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

from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import (
    Updater, CommandHandler, CallbackQueryHandler, MessageHandler,
    Filters, CallbackContext, ConversationHandler
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
        
        # Updater und Dispatcher
        try:
            self.updater = Updater(self.token, use_context=True)
            self.dispatcher = self.updater.dispatcher
            self.bot = self.updater.bot
            
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
        self.dispatcher.add_handler(CommandHandler("start", self._cmd_welcome))
        self.dispatcher.add_handler(CommandHandler("help", self._cmd_help))
        
        # Bot-Steuerungsbefehle
        self.dispatcher.add_handler(CommandHandler("status", self._cmd_status))
        self.dispatcher.add_handler(CommandHandler("startbot", self._cmd_start_bot))
        self.dispatcher.add_handler(CommandHandler("stopbot", self._cmd_stop_bot))
        self.dispatcher.add_handler(CommandHandler("pausebot", self._cmd_pause_bot))
        self.dispatcher.add_handler(CommandHandler("resumebot", self._cmd_resume_bot))
        
        # Trading-Befehle
        self.dispatcher.add_handler(CommandHandler("balance", self._cmd_balance))
        self.dispatcher.add_handler(CommandHandler("positions", self._cmd_positions))
        self.dispatcher.add_handler(CommandHandler("orders", self._cmd_orders))
        
        # Analyse-Befehle
        self.dispatcher.add_handler(CommandHandler("performance", self._cmd_performance))
        self.dispatcher.add_handler(CommandHandler("market", self._cmd_market))
        self.dispatcher.add_handler(CommandHandler("prediction", self._cmd_prediction))
        
        # Callback-Handler für Buttons
        self.dispatcher.add_handler(CallbackQueryHandler(self._handle_button_press))
        
        # Fehlerhandler
        self.dispatcher.add_error_handler(self._error_handler)
    
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
            self.updater.start_polling()
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
            self.updater.stop()
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
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = ParseMode.HTML,
                     reply_markup: Any = None, disable_notification: bool = False) -> Optional[Any]:
        """Sendet eine Nachricht an einen Chat."""
        try:
            return self.bot.send_message(
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
    
    def _handle_button_press(self, update: Update, context: CallbackContext):
        """Verarbeitet Klicks auf Inline-Buttons."""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Autorisierung prüfen
        if not self._is_user_authorized(user_id):
            query.answer("❌ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        try:
            # Callback-Daten extrahieren
            callback_data = query.data
            
            # Bestätigung anzeigen
            query.answer()
            
            # Callback-Typen verarbeiten
            if callback_data == "menu":
                # Hauptmenü anzeigen
                self._show_main_menu(query.message)
            elif callback_data == "status":
                # Status abrufen (verzögerte Antwort über Edit)
                query.edit_message_text("⏳ Aktualisiere Status...")
                self._cmd_status(update, context)
            elif callback_data == "balance":
                # Kontostand abrufen
                query.edit_message_text("⏳ Rufe Kontostand ab...")
                self._cmd_balance(update, context)
            elif callback_data == "positions":
                # Positionen abrufen
                query.edit_message_text("⏳ Rufe Positionen ab...")
                self._cmd_positions(update, context)
            elif callback_data == "orders":
                # Orders abrufen
                query.edit_message_text("⏳ Rufe Orders ab...")
                self._cmd_orders(update, context)
            elif callback_data == "performance":
                # Performance abrufen
                query.edit_message_text("⏳ Rufe Performance-Daten ab...")
                self._cmd_performance(update, context)
            elif callback_data == "startbot":
                # Bot starten
                query.edit_message_text("⏳ Starte Bot...")
                self._cmd_start_bot(update, context)
            elif callback_data == "stopbot":
                # Bot stoppen
                query.edit_message_text("⏳ Stoppe Bot...")
                self._cmd_stop_bot(update, context)
            elif callback_data == "pausebot":
                # Bot pausieren
                query.edit_message_text("⏳ Pausiere Bot...")
                self._cmd_pause_bot(update, context)
            elif callback_data == "resumebot":
                # Bot fortsetzen
                query.edit_message_text("⏳ Setze Bot fort...")
                self._cmd_resume_bot(update, context)
            elif callback_data == "market":
                # Marktdaten abrufen
                query.edit_message_text("⏳ Rufe Marktdaten ab...")
                self._cmd_market(update, context)
            elif callback_data == "prediction":
                # Prognose abrufen
                query.edit_message_text("⏳ Rufe Prognose ab...")
                self._cmd_prediction(update, context)
            else:
                # Unbekannter Callback
                query.edit_message_text(f"Unbekannte Aktion: {callback_data}")
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Button-Klicks: {str(e)}")
            try:
                query.edit_message_text(f"❌ Fehler: {str(e)}")
            except:
                pass
    
    # Command-Handler
    
    def _cmd_welcome(self, update: Update, context: CallbackContext):
        """Begrüßt den Benutzer und zeigt Hauptmenü an."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
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
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe Performance-Daten ab...")
        
        try:
            # Führe den performance-Befehl aus, falls registriert
            if 'performance' in self.commands:
                performance = self.commands['performance']()
                
                if performance:
                    # Performance-Daten formatieren
                    total_trades = performance.get('total_trades', 0)
                    winning_trades = performance.get('winning_trades', 0)
                    losing_trades = performance.get('losing_trades', 0)
                    win_rate = performance.get('win_rate', 0) * 100
                    
                    profit_today = performance.get('profit_today', 0)
                    profit_week = performance.get('profit_week', 0)
                    profit_month = performance.get('profit_month', 0)
                    profit_total = performance.get('profit_total', 0)
                    
                    # Weitere Metriken
                    avg_profit = performance.get('avg_profit', 0)
                    avg_loss = performance.get('avg_loss', 0)
                    max_drawdown = performance.get('max_drawdown', 0) * 100
                    sharpe_ratio = performance.get('sharpe_ratio', 0)
                    
                    # Antwort-Text erstellen
                    performance_text = (
                        f"📊 <b>Performance-Metriken</b>\n\n"
                        
                        f"<b>Handelsergebnisse:</b>\n"
                        f"Trades gesamt: {total_trades}\n"
                        f"Gewinne: {winning_trades}\n"
                        f"Verluste: {losing_trades}\n"
                        f"Win-Rate: {win_rate:.2f}%\n\n"
                        
                        f"<b>Profitabilität:</b>\n"
                        f"Heute: {profit_today:.2f} USDT\n"
                        f"Diese Woche: {profit_week:.2f} USDT\n"
                        f"Diesen Monat: {profit_month:.2f} USDT\n"
                        f"Gesamt: {profit_total:.2f} USDT\n\n"
                        
                        f"<b>Risikometriken:</b>\n"
                        f"Ø Gewinn: {avg_profit:.2f} USDT\n"
                        f"Ø Verlust: {avg_loss:.2f} USDT\n"
                        f"Max. Drawdown: {max_drawdown:.2f}%\n"
                        f"Sharpe Ratio: {sharpe_ratio:.2f}\n\n"
                        
                        f"<i>Stand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</i>"
                    )
                    
                    # Buttons für weitere Aktionen
                    keyboard = [
                        [
                            InlineKeyboardButton("🔄 Aktualisieren", callback_data="performance"),
                            InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                        ],
                        [
                            InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                            InlineKeyboardButton("💰 Kontostand", callback_data="balance")
                        ]
                    ]                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    message.edit_text(
                        performance_text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup
                    )
                else:
                    message.edit_text("❌ Konnte Performance-Daten nicht abrufen.")
            else:
                message.edit_text("❌ Performance-Befehl nicht verfügbar.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Daten: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_market(self, update: Update, context: CallbackContext):
        """Zeigt Marktanalyse-Daten."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe Marktdaten ab...")
        
        try:
            # Marktdaten-Handler, falls vorhanden
            market_data = None
            if 'market' in self.commands:
                market_data = self.commands['market']()
            
            if market_data:
                # Marktdaten formatieren
                market_text = "<b>🌍 Marktübersicht</b>\n\n"
                
                # Globale Markt-Stimmung
                sentiment = market_data.get('sentiment', {})
                sentiment_score = sentiment.get('score', 0)
                sentiment_label = sentiment.get('label', 'Neutral')
                
                sentiment_emoji = {
                    'BULLISH': '🟢',
                    'SLIGHTLY_BULLISH': '🟢',
                    'NEUTRAL': '⚪',
                    'SLIGHTLY_BEARISH': '🔴',
                    'BEARISH': '🔴'
                }.get(sentiment_label, '⚪')
                
                market_text += f"<b>Markt-Stimmung:</b> {sentiment_emoji} {sentiment_label}\n\n"
                
                # Top Coins nach Performance
                top_coins = market_data.get('top_performers', [])
                if top_coins:
                    market_text += "<b>Top Performer (24h):</b>\n"
                    for coin in top_coins[:5]:
                        market_text += f"- {coin['symbol']}: {coin['change_24h']:.2f}%\n"
                    market_text += "\n"
                
                # Schlechteste Coins nach Performance
                worst_coins = market_data.get('worst_performers', [])
                if worst_coins:
                    market_text += "<b>Schlechteste Performer (24h):</b>\n"
                    for coin in worst_coins[:5]:
                        market_text += f"- {coin['symbol']}: {coin['change_24h']:.2f}%\n"
                    market_text += "\n"
                
                # Markt-Kapitaliserung und Volumen
                market_cap = market_data.get('total_market_cap', 0)
                volume_24h = market_data.get('total_volume_24h', 0)
                btc_dominance = market_data.get('btc_dominance', 0)
                
                market_text += (
                    f"<b>Markt-Kennzahlen:</b>\n"
                    f"Gesamtmarkt-Kapitalisierung: ${market_cap:,.0f}\n"
                    f"24h-Handelsvolumen: ${volume_24h:,.0f}\n"
                    f"BTC-Dominanz: {btc_dominance:.2f}%\n\n"
                )
                
                # Fear & Greed Index
                fear_greed = market_data.get('fear_greed_index', {})
                if fear_greed:
                    value = fear_greed.get('value', 0)
                    label = fear_greed.get('label', 'Neutral')
                    
                    # Emoji basierend auf Fear & Greed
                    if value <= 25:
                        fng_emoji = '😨'  # Extreme Fear
                    elif value <= 40:
                        fng_emoji = '😰'  # Fear
                    elif value <= 60:
                        fng_emoji = '😐'  # Neutral
                    elif value <= 75:
                        fng_emoji = '😊'  # Greed
                    else:
                        fng_emoji = '🤑'  # Extreme Greed
                    
                    market_text += f"<b>Fear & Greed Index:</b> {fng_emoji} {value} ({label})\n\n"
                
                market_text += f"<i>Stand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</i>"
                
                # Buttons für weitere Aktionen
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Aktualisieren", callback_data="market"),
                        InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                    ],
                    [
                        InlineKeyboardButton("🔮 Prognose", callback_data="prediction"),
                        InlineKeyboardButton("📈 Positionen", callback_data="positions")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message.edit_text(
                    market_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup
                )
            else:
                message.edit_text("❌ Konnte Marktdaten nicht abrufen.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Marktdaten: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
    def _cmd_prediction(self, update: Update, context: CallbackContext):
        """Zeigt aktuelle Prognosen."""
        user_id = update.effective_user.id
        
        if not self._is_user_authorized(user_id):
            update.message.reply_text("⛔ Sie sind nicht autorisiert, diesen Bot zu verwenden.")
            return
        
        if not self.controller:
            update.message.reply_text("❌ Controller-Referenz fehlt!")
            return
        
        message = update.effective_message.reply_text("⏳ Rufe Prognosen ab...")
        
        try:
            # Prognosedaten-Handler, falls vorhanden
            prediction_data = None
            if 'prediction' in self.commands:
                prediction_data = self.commands['prediction']()
            
            if prediction_data:
                # Prognosedaten formatieren
                predictions_text = "<b>🔮 Aktuelle Prognosen</b>\n\n"
                
                # Einzelne Asset-Prognosen
                assets = prediction_data.get('assets', [])
                if assets:
                    for asset in assets:
                        symbol = asset.get('symbol', 'Unbekannt')
                        direction = asset.get('direction', 'Neutral')
                        confidence = asset.get('confidence', 0) * 100
                        timeframe = asset.get('timeframe', '1h')
                        
                        # Emoji basierend auf Richtung
                        direction_emoji = '🟢' if direction == 'up' else '🔴' if direction == 'down' else '⚪'
                        
                        predictions_text += (
                            f"{direction_emoji} <b>{symbol}</b> ({timeframe})\n"
                            f"Richtung: {direction.upper()}\n"
                            f"Konfidenz: {confidence:.1f}%\n\n"
                        )
                else:
                    predictions_text += "Keine Asset-Prognosen verfügbar.\n\n"
                
                # Gesamtmarkt-Prognose
                market_prediction = prediction_data.get('market', {})
                if market_prediction:
                    market_direction = market_prediction.get('direction', 'neutral')
                    market_confidence = market_prediction.get('confidence', 0) * 100
                    
                    # Emoji basierend auf Marktrichtung
                    market_emoji = '🟢' if market_direction == 'bullish' else '🔴' if market_direction == 'bearish' else '⚪'
                    
                    predictions_text += (
                        f"<b>Gesamtmarkt-Prognose:</b>\n"
                        f"{market_emoji} Richtung: {market_direction.upper()}\n"
                        f"Konfidenz: {market_confidence:.1f}%\n\n"
                    )
                
                predictions_text += f"<i>Stand: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</i>"
                
                # Buttons für weitere Aktionen
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Aktualisieren", callback_data="prediction"),
                        InlineKeyboardButton("🔙 Hauptmenü", callback_data="menu")
                    ],
                    [
                        InlineKeyboardButton("🌍 Markt", callback_data="market"),
                        InlineKeyboardButton("📈 Positionen", callback_data="positions")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message.edit_text(
                    predictions_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup
                )
            else:
                message.edit_text("❌ Konnte Prognosen nicht abrufen.")
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Prognosen: {str(e)}")
            message.edit_text(f"❌ Fehler: {str(e)}")
    
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
    
    # Utility-Methoden für den Controller
    
    def _get_account_balance(self):
        """
        Hilfsmethode für den Controller zum Abrufen des Kontostands.
        
        Returns:
            Balance-Objekt oder None bei Fehler
        """
        if not self.controller:
            self.logger.error("Controller-Referenz fehlt für _get_account_balance")
            return None
        
        try:
            # Live Trading Modul finden
            if hasattr(self.controller, 'modules') and 'live_trading' in self.controller.modules:
                live_trading = self.controller.modules['live_trading']
                if hasattr(live_trading, 'get_account_balance'):
                    return live_trading.get_account_balance()
            
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            return None
    
    def _get_open_positions(self):
        """
        Hilfsmethode für den Controller zum Abrufen der offenen Positionen.
        
        Returns:
            Liste der offenen Positionen oder None bei Fehler
        """
        if not self.controller:
            self.logger.error("Controller-Referenz fehlt für _get_open_positions")
            return None
        
        try:
            # Live Trading Modul finden
            if hasattr(self.controller, 'modules') and 'live_trading' in self.controller.modules:
                live_trading = self.controller.modules['live_trading']
                if hasattr(live_trading, 'get_open_positions'):
                    return live_trading.get_open_positions()
            
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der offenen Positionen: {str(e)}")
            return None
    
    def _get_performance_metrics(self):
        """
        Hilfsmethode für den Controller zum Abrufen der Performance-Metriken.
        
        Returns:
            Performance-Metriken oder None bei Fehler
        """
        if not self.controller:
            self.logger.error("Controller-Referenz fehlt für _get_performance_metrics")
            return None
        
        try:
            # Learning Module finden
            if hasattr(self.controller, 'modules') and 'learning_module' in self.controller.modules:
                learning_module = self.controller.modules['learning_module']
                if hasattr(learning_module, 'get_performance_metrics'):
                    return learning_module.get_performance_metrics()
            
            # Alternativ aus dem Tax Module
            if hasattr(self.controller, 'modules') and 'tax_module' in self.controller.modules:
                tax_module = self.controller.modules['tax_module']
                if hasattr(tax_module, 'get_performance_metrics'):
                    return tax_module.get_performance_metrics()
            
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Metriken: {str(e)}")
            return None
    
    def _process_transcript_command(self, path: str = None):
        """
        Hilfsmethode für den Controller zum Verarbeiten eines Transkripts.
        
        Args:
            path: Pfad zum Transkript (optional)
            
        Returns:
            Ergebnis der Transkriptverarbeitung oder None bei Fehler
        """
        if not self.controller:
            self.logger.error("Controller-Referenz fehlt für _process_transcript_command")
            return None
        
        if not path:
            self.logger.error("Kein Pfad für die Transkriptverarbeitung angegeben")
            return None
        
        try:
            if hasattr(self.controller, 'process_transcript'):
                return self.controller.process_transcript(path)
            
            return None
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Transkripts: {str(e)}")
            return None

# telegram_module.py

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Union

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

class TelegramBot:
    """
    Ein Telegram-Bot mit detailliertem Logging für Fehlersuche.
    Protokolliert jede Benutzerinteraktion in Echtzeit auf der Konsole.
    """

    def __init__(self, token: str, admin_ids: List[int] = None, log_level=logging.INFO):
        """
        Initialisiert den TelegramBot.
        
        Args:
            token: Telegram Bot API Token
            admin_ids: Liste von Telegram-Benutzer-IDs mit Admin-Rechten
            log_level: Logging-Level (Default: INFO)
        """
        # Logger konfigurieren
        self.logger = logging.getLogger("TelegramBot")
        self.logger.setLevel(log_level)
        self.admin_ids = admin_ids or []
        
        # Bot initialisieren
        self.logger.info("=== TELEGRAM BOT INITIALISIERUNG STARTET ===")
        self.application = Application.builder().token(token).build()
        
        # Handler registrieren
        self._register_handlers()
        
        # Callback-Funktionen-Dictionary
        self.command_handlers = {}
        self.button_handlers = {}
        
        self.logger.info("=== TELEGRAM BOT INITIALISIERUNG ABGESCHLOSSEN ===")
    
    def _register_handlers(self):
        """Registriert alle Handler für den Bot"""
        self.logger.info("Registriere Telegram-Handler...")
        
        # Befehle
        self.application.add_handler(CommandHandler("start", self._log_and_handle_start))
        self.application.add_handler(CommandHandler("help", self._log_and_handle_help))
        self.application.add_handler(CommandHandler("status", self._log_and_handle_status))
        self.application.add_handler(CommandHandler("settings", self._log_and_handle_settings))
        
        # Button-Callbacks
        self.application.add_handler(CallbackQueryHandler(self._log_and_handle_button))
        
        # Textnachrichten
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._log_and_handle_message
        ))
        
        # Fehlerhandler
        self.application.add_error_handler(self._handle_error)
        
        self.logger.info("Alle Handler erfolgreich registriert")
    
    def _log_user_action(self, update: Update, action_type: str, details: Dict[str, Any] = None):
        """
        Erstellt einen detaillierten Log-Eintrag für eine Benutzeraktion.
        
        Args:
            update: Das Update-Objekt von Telegram
            action_type: Art der Aktion (Befehl, Button, etc.)
            details: Zusätzliche Details zur Aktion
        """
        user = update.effective_user
        chat = update.effective_chat
        
        # Basisinformationen
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "action": action_type,
            "user_id": user.id if user else "Unbekannt",
            "username": user.username if user else "Unbekannt",
            "first_name": user.first_name if user else "Unbekannt",
            "chat_id": chat.id if chat else "Unbekannt",
            "chat_type": chat.type if chat else "Unbekannt",
        }
        
        # Zusätzliche Details hinzufügen
        if details:
            log_data.update(details)
        
        # Log formatieren
        log_msg = f"[{log_data['timestamp']}] {action_type.upper()} | "
        log_msg += f"Benutzer: {log_data['user_id']} (@{log_data['username']}) | "
        log_msg += f"Chat: {log_data['chat_id']} ({log_data['chat_type']})"
        
        # Details ausgeben
        if details:
            log_msg += " | Details: " + " | ".join(f"{k}={v}" for k, v in details.items())
        
        # Log ausgeben mit visueller Trennung je nach Aktionstyp
        if action_type == "button":
            self.logger.info(f"🔘 {log_msg}")
        elif action_type == "command":
            self.logger.info(f"🔹 {log_msg}")
        elif action_type == "message":
            self.logger.info(f"💬 {log_msg}")
        elif action_type == "error":
            self.logger.error(f"❌ {log_msg}")
        else:
            self.logger.info(f"ℹ️ {log_msg}")
        
        return log_data
    
    async def _log_and_handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet den /start Befehl"""
        self._log_user_action(update, "command", {"command": "/start"})
        
        # Haupt-Menü anzeigen
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("💰 Handeln", callback_data="trade")
            ],
            [
                InlineKeyboardButton("⚙️ Einstellungen", callback_data="settings"),
                InlineKeyboardButton("📈 Status", callback_data="status")
            ],
            [
                InlineKeyboardButton("📰 News", callback_data="news"),
                InlineKeyboardButton("🔍 Info", callback_data="info")
            ]
        ]
        
        try:
            await update.message.reply_text(
                "Willkommen beim Trading Bot! Wähle eine Option:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            self.logger.debug(f"Start-Menü an Benutzer {update.effective_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "start_command",
                "details": traceback.format_exc()
            })
    
    async def _log_and_handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet den /help Befehl"""
        self._log_user_action(update, "command", {"command": "/help"})
        
        help_text = (
            "📚 *Verfügbare Befehle:*\n"
            "/start - Startet den Bot und zeigt das Hauptmenü\n"
            "/help - Zeigt diese Hilfe an\n"
            "/status - Zeigt den aktuellen Status\n"
            "/settings - Zeigt Einstellungsoptionen\n\n"
            "Du kannst auch die Buttons im Menü verwenden, um zu navigieren."
        )
        
        try:
            await update.message.reply_text(help_text, parse_mode="Markdown")
            self.logger.debug(f"Hilfe-Text an Benutzer {update.effective_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "help_command"
            })
    
    async def _log_and_handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet den /status Befehl"""
        self._log_user_action(update, "command", {"command": "/status"})
        
        # Hier könnte man Statusinformationen aus anderen Modulen einbinden
        status_text = (
            "🔄 *Aktueller Bot-Status:*\n"
            "• Bot ist aktiv und läuft\n"
            "• Verbunden mit Telegram API\n"
            "• Server-Zeit: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            "• Version: 1.0.0\n"
        )
        
        try:
            await update.message.reply_text(status_text, parse_mode="Markdown")
            self.logger.debug(f"Status-Text an Benutzer {update.effective_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "status_command"
            })
    
    async def _log_and_handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet den /settings Befehl"""
        self._log_user_action(update, "command", {"command": "/settings"})
        
        # Einstellungs-Menü anzeigen
        keyboard = [
            [
                InlineKeyboardButton("🔔 Benachrichtigungen", callback_data="settings_notifications"),
                InlineKeyboardButton("💲 Währungen", callback_data="settings_currencies")
            ],
            [
                InlineKeyboardButton("🔐 Sicherheit", callback_data="settings_security"),
                InlineKeyboardButton("⏱️ Zeitrahmen", callback_data="settings_timeframes")
            ],
            [
                InlineKeyboardButton("🔙 Zurück zum Hauptmenü", callback_data="back_to_main")
            ]
        ]
        
        try:
            await update.message.reply_text(
                "⚙️ *Einstellungen:*\n"
                "Wähle eine Kategorie, um Einstellungen anzupassen.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Einstellungs-Menü an Benutzer {update.effective_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "settings_command"
            })
    
    async def _log_and_handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet Button-Klicks"""
        query = update.callback_query
        
        # Detailliertes Logging
        self._log_user_action(update, "button", {
            "callback_data": query.data,
            "message_id": query.message.message_id
        })
        
        try:
            # Bestätige den Button-Klick gegenüber Telegram
            await query.answer()
            self.logger.debug(f"Button-Klick bestätigt: {query.data}")
            
            # Verarbeitung der verschiedenen Button-Aktionen
            if query.data == "portfolio":
                await self._handle_portfolio_button(update, context)
            elif query.data == "trade":
                await self._handle_trade_button(update, context)
            elif query.data == "settings":
                await self._handle_settings_button(update, context)
            elif query.data == "status":
                await self._handle_status_button(update, context)
            elif query.data == "news":
                await self._handle_news_button(update, context)
            elif query.data == "info":
                await self._handle_info_button(update, context)
            elif query.data == "back_to_main":
                await self._handle_back_to_main_button(update, context)
            elif query.data.startswith("settings_"):
                await self._handle_settings_subcategory(update, context, query.data)
            else:
                # Für unbekannte Button-Daten
                self.logger.warning(f"Unbehandelte Button-Daten: {query.data}")
                # Benutzerdefinierten Handler aufrufen, falls registriert
                if query.data in self.button_handlers:
                    handler_func = self.button_handlers[query.data]
                    await handler_func(update, context)
        
        except Exception as e:
            error_details = {
                "error": str(e),
                "location": "button_handler",
                "callback_data": query.data,
                "traceback": traceback.format_exc()
            }
            self._log_user_action(update, "error", error_details)
            
            try:
                # Informiere den Benutzer über den Fehler
                await query.message.reply_text(
                    "❌ Bei der Verarbeitung deiner Anfrage ist ein Fehler aufgetreten. "
                    "Bitte versuche es später erneut."
                )
            except:
                pass  # Wenn selbst die Fehlermeldung nicht gesendet werden kann
    
    async def _log_and_handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Protokolliert und verarbeitet Textnachrichten"""
        self._log_user_action(update, "message", {"text": update.message.text})
        
        # Hier könnte eine Verarbeitung von Textnachrichten implementiert werden
        # Zum Beispiel eine einfache Antwort
        try:
            await update.message.reply_text(
                "Ich verstehe nur Befehle und Buttons. Versuche /help für eine Liste von Befehlen."
            )
            self.logger.debug(f"Standardantwort auf Textnachricht an Benutzer {update.effective_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "message_handler"
            })
    
    async def _handle_portfolio_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Portfolio-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Portfolio-Button für Benutzer {query.from_user.id}")
        
        # Hier würde man Portfoliodaten aus anderen Modulen abrufen
        portfolio_text = (
            "📊 *Dein Portfolio:*\n\n"
            "• BTC: 0.05 BTC (1,500.00 EUR)\n"
            "• ETH: 0.5 ETH (750.00 EUR)\n"
            "• SOL: 10 SOL (500.00 EUR)\n\n"
            "Gesamtwert: 2,750.00 EUR\n"
            "24h Änderung: +3.5%"
        )
        
        try:
            # Zurück-Button
            keyboard = [[InlineKeyboardButton("🔙 Zurück", callback_data="back_to_main")]]
            
            await query.message.edit_text(
                portfolio_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Portfolio-Daten an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "portfolio_button_handler"
            })
    
    async def _handle_trade_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Handels-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Trade-Button für Benutzer {query.from_user.id}")
        
        keyboard = [
            [
                InlineKeyboardButton("🟢 Kaufen", callback_data="trade_buy"),
                InlineKeyboardButton("🔴 Verkaufen", callback_data="trade_sell")
            ],
            [InlineKeyboardButton("🔙 Zurück", callback_data="back_to_main")]
        ]
        
        try:
            await query.message.edit_text(
                "💰 *Handelsoptionen:*\n"
                "Wähle eine Aktion aus:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Handelsoptionen an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "trade_button_handler"
            })
    
    async def _handle_settings_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Einstellungs-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Settings-Button für Benutzer {query.from_user.id}")
        
        # Gleicher Inhalt wie bei /settings Befehl
        keyboard = [
            [
                InlineKeyboardButton("🔔 Benachrichtigungen", callback_data="settings_notifications"),
                InlineKeyboardButton("💲 Währungen", callback_data="settings_currencies")
            ],
            [
                InlineKeyboardButton("🔐 Sicherheit", callback_data="settings_security"),
                InlineKeyboardButton("⏱️ Zeitrahmen", callback_data="settings_timeframes")
            ],
            [
                InlineKeyboardButton("🔙 Zurück zum Hauptmenü", callback_data="back_to_main")
            ]
        ]
        
        try:
            await query.message.edit_text(
                "⚙️ *Einstellungen:*\n"
                "Wähle eine Kategorie, um Einstellungen anzupassen.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Einstellungs-Menü an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "settings_button_handler"
            })
    
    async def _handle_status_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Status-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Status-Button für Benutzer {query.from_user.id}")
        
        # Ähnlicher Inhalt wie beim /status Befehl
        status_text = (
            "🔄 *Aktueller Bot-Status:*\n\n"
            "• Bot ist aktiv und läuft\n"
            "• Verbunden mit Telegram API\n"
            "• Server-Zeit: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            "• Version: 1.0.0\n\n"
            "🖥️ *Server-Status:*\n"
            "• CPU: 25%\n"
            "• RAM: 512MB/2GB\n"
            "• Laufzeit: 3d 12h 45m\n"
        )
        
        keyboard = [[InlineKeyboardButton("🔙 Zurück", callback_data="back_to_main")]]
        
        try:
            await query.message.edit_text(
                status_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Status-Information an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "status_button_handler"
            })
    
    async def _handle_news_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den News-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite News-Button für Benutzer {query.from_user.id}")
        
        news_text = (
            "📰 *Aktuelle Krypto-News:*\n\n"
            "1. Bitcoin durchbricht wichtige Widerstandszone\n"
            "2. Neue Regulierungen in der EU angekündigt\n"
            "3. Ethereum-Entwickler planen Update für Q2\n"
            "4. Großer Fonds investiert in Solana-Ökosystem\n\n"
            "Letzte Aktualisierung: " + datetime.now().strftime("%H:%M:%S")
        )
        
        keyboard = [[InlineKeyboardButton("🔙 Zurück", callback_data="back_to_main")]]
        
        try:
            await query.message.edit_text(
                news_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"News-Information an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "news_button_handler"
            })
    
    async def _handle_info_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Info-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Info-Button für Benutzer {query.from_user.id}")
        
        info_text = (
            "🔍 *Bot-Informationen:*\n\n"
            "Dieser Trading-Bot wurde entwickelt, um den Handel mit Kryptowährungen "
            "zu automatisieren und zu vereinfachen.\n\n"
            "Version: 1.0.0\n"
            "Entwickler: Trading Bot Team\n"
            "Kontakt: support@example.com"
        )
        
        keyboard = [[InlineKeyboardButton("🔙 Zurück", callback_data="back_to_main")]]
        
        try:
            await query.message.edit_text(
                info_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Info-Text an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "info_button_handler"
            })
    
    async def _handle_back_to_main_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Zurück-zum-Hauptmenü-Button"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Back-to-Main-Button für Benutzer {query.from_user.id}")
        
        # Hauptmenü erneut anzeigen
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("💰 Handeln", callback_data="trade")
            ],
            [
                InlineKeyboardButton("⚙️ Einstellungen", callback_data="settings"),
                InlineKeyboardButton("📈 Status", callback_data="status")
            ],
            [
                InlineKeyboardButton("📰 News", callback_data="news"),
                InlineKeyboardButton("🔍 Info", callback_data="info")
            ]
        ]
        
        try:
            await query.message.edit_text(
                "Hauptmenü des Trading Bots. Wähle eine Option:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            self.logger.debug(f"Hauptmenü an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "back_to_main_button_handler"
            })
    
    async def _handle_settings_subcategory(self, update: Update, context: ContextTypes.DEFAULT_TYPE, subcategory: str):
        """Verarbeitet Einstellungs-Unterkategorien"""
        query = update.callback_query
        self.logger.info(f"Verarbeite Settings-Subkategorie {subcategory} für Benutzer {query.from_user.id}")
        
        # Je nach Unterkategorie unterschiedliche Inhalte anzeigen
        if subcategory == "settings_notifications":
            text = "🔔 *Benachrichtigungseinstellungen:*\n\nHier können Sie Ihre Benachrichtigungspräferenzen anpassen."
        elif subcategory == "settings_currencies":
            text = "💲 *Währungseinstellungen:*\n\nHier können Sie die zu überwachenden Währungen anpassen."
        elif subcategory == "settings_security":
            text = "🔐 *Sicherheitseinstellungen:*\n\nHier können Sie Sicherheitsoptionen konfigurieren."
        elif subcategory == "settings_timeframes":
            text = "⏱️ *Zeitrahmeneinstellungen:*\n\nHier können Sie die Analyse-Zeitrahmen anpassen."
        else:
            text = "⚙️ Einstellungen"
        
        # Zurück zu Einstellungen Button
        keyboard = [[InlineKeyboardButton("🔙 Zurück zu Einstellungen", callback_data="settings")]]
        
        try:
            await query.message.edit_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.debug(f"Einstellungs-Unterkategorie {subcategory} an Benutzer {query.from_user.id} gesendet")
        except Exception as e:
            self._log_user_action(update, "error", {
                "error": str(e),
                "location": "settings_subcategory_handler",
                "subcategory": subcategory
            })
    
    async def _handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Behandelt Fehler im Bot"""
        if update:
            chat_id = update.effective_chat.id if update.effective_chat else "Unbekannt"
            user_id = update.effective_user.id if update.effective_user else "Unbekannt"
        else:
            chat_id = "Unbekannt"
            user_id = "Unbekannt"
        
        error_msg = f"Fehler beim Update: Chat {chat_id}, Benutzer {user_id}"
        if context.error:
            error_msg += f", Fehler: {context.error}"
        
        self.logger.error(f"❌ {error_msg}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Optional: Benachrichtige Admins über Fehler
        for admin_id in self.admin_ids:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=f"⚠️ *Bot-Fehler*\n\n{error_msg}\n\nZeitstempel: {datetime.now()}",
                    parse_mode="Markdown"
                )
            except:
                self.logger.error(f"Konnte Admin {admin_id} nicht über Fehler informieren")
    
    def register_command_handler(self, command: str, handler_func: Callable):
        """
        Registriert eine benutzerdefinierte Handler-Funktion für einen Befehl.
        
        Args:
            command: Befehl ohne / (z.B. 'mycommand')
            handler_func: Async-Funktion, die den Befehl verarbeitet
        """
        self.logger.info(f"Registriere benutzerdefinierten Handler für Befehl /{command}")
        self.command_handlers[command] = handler_func
        self.application.add_handler(CommandHandler(command, handler_func))
    
    def register_button_handler(self, callback_data: str, handler_func: Callable):
        """
        Registriert eine benutzerdefinierte Handler-Funktion für einen Button.
        
        Args:
            callback_data: Callback-Daten des Buttons
            handler_func: Async-Funktion, die den Button-Klick verarbeitet
        """
        self.logger.info(f"Registriere benutzerdefinierten Handler für Button {callback_data}")
        self.button_handlers[callback_data] = handler_func
    
    def run(self, webhook_url: str = None):
        """
        Startet den Bot entweder im Polling-Modus oder im Webhook-Modus.
        
        Args:
            webhook_url: URL für Webhook, falls Webhook-Modus verwendet werden soll
        """
        if webhook_url:
            self.logger.info(f"Starte Bot im Webhook-Modus mit URL: {webhook_url}")
            domain = webhook_url.split("://")[1].split("/")[0]
            self.application.run_webhook(
                listen="0.0.0.0",
                port=8443,
                url_path=webhook_url.split(domain)[1],
                webhook_url=webhook_url
            )
        else:
            self.logger.info("Starte Bot im Polling-Modus")
            self.application.run_polling()

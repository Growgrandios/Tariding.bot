# telegram_module.py

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Union

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
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
    Telegram-Bot mit ausführlichem Logging für Debugging-Zwecke.
    Protokolliert jede Interaktion in der Konsole, um Button-Probleme zu identifizieren.
    """

    def __init__(self, token: str, admin_ids: List[int] = None):
        """
        Initialisiert den TelegramBot.
        
        Args:
            token: Telegram Bot API Token
            admin_ids: Liste von Telegram-Benutzer-IDs mit Admin-Rechten
        """
        # Logger einrichten
        self.logger = self.setup_logger()
        self.logger.info("========== TELEGRAM BOT INITIALISIERUNG ==========")
        
        # Bot-Konfiguration
        self.token = token
        self.admin_ids = admin_ids or []
        
        # Anwendung erstellen
        self.logger.info(f"Erstelle Telegram-Bot mit Token: {token[:5]}...{token[-5:] if len(token) > 10 else ''}")
        try:
            self.application = Application.builder().token(token).build()
            self.logger.info("Telegram-Application erfolgreich erstellt")
        except Exception as e:
            self.logger.critical(f"FEHLER beim Erstellen der Telegram-Application: {str(e)}")
            self.logger.critical(traceback.format_exc())
            raise

        # Handler registrieren
        self._register_handlers()
        self.logger.info("========== TELEGRAM BOT BEREIT ==========")
    
    def setup_logger(self):
        """Richtet den Logger für detaillierte Konsolenausgabe ein"""
        logger = logging.getLogger("TelegramBot")
        logger.setLevel(logging.DEBUG)
        
        # Handler für Konsolenausgabe
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Format für die Logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Handler hinzufügen
        logger.addHandler(console_handler)
        return logger
    
    def _register_handlers(self):
        """Registriert alle Handler für den Bot"""
        self.logger.info("Registriere Telegram-Handler...")
        
        try:
            # Befehle
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(CommandHandler("help", self._help_command))
            self.application.add_handler(CommandHandler("status", self._status_command))
            
            # Button-Callbacks - DAS WICHTIGSTE FÜR DIE FEHLERBEHEBUNG
            self.logger.info("Registriere CallbackQueryHandler für Button-Interaktionen")
            self.application.add_handler(CallbackQueryHandler(self._button_callback))
            
            # Textnachrichten
            self.application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, self._message_handler
            ))
            
            # Fehlerhandler
            self.application.add_error_handler(self._error_handler)
            
            self.logger.info("Alle Handler erfolgreich registriert")
        except Exception as e:
            self.logger.critical(f"FEHLER bei der Handler-Registrierung: {str(e)}")
            self.logger.critical(traceback.format_exc())
            raise
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /start Befehl"""
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Ausführliches Logging
        self.logger.info(f"Start-Befehl empfangen | User: {user.id} (@{user.username}) | Chat: {chat_id}")
        
        # Inline-Keyboard für Hauptmenü
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("⚙️ Einstellungen", callback_data="settings")
            ],
            [
                InlineKeyboardButton("📈 Status", callback_data="status"),
                InlineKeyboardButton("🔍 Info", callback_data="info")
            ]
        ]
        
        try:
            # Nachricht senden
            self.logger.debug(f"Sende Start-Menü an User {user.id}...")
            await update.message.reply_text(
                f"Willkommen beim Trading Bot, {user.first_name}!\n\n"
                "Wähle eine Option aus dem Menü:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            self.logger.info(f"Start-Menü erfolgreich gesendet | User: {user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Senden des Start-Menüs: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /help Befehl"""
        user = update.effective_user
        self.logger.info(f"Help-Befehl empfangen | User: {user.id} (@{user.username})")
        
        try:
            await update.message.reply_text(
                "📚 *Verfügbare Befehle:*\n"
                "/start - Startet den Bot und zeigt das Hauptmenü\n"
                "/help - Zeigt diese Hilfe an\n"
                "/status - Zeigt den aktuellen Status\n\n"
                "Du kannst auch die Buttons im Menü verwenden.",
                parse_mode="Markdown"
            )
            self.logger.info(f"Hilfe-Text erfolgreich gesendet | User: {user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Senden der Hilfe: {str(e)}")
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für den /status Befehl"""
        user = update.effective_user
        self.logger.info(f"Status-Befehl empfangen | User: {user.id} (@{user.username})")
        
        try:
            await update.message.reply_text(
                "🔄 *Bot-Status:*\n"
                "• Bot ist aktiv und läuft\n"
                "• Verbunden mit Telegram API\n"
                f"• Server-Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                parse_mode="Markdown"
            )
            self.logger.info(f"Status-Info erfolgreich gesendet | User: {user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Senden des Status: {str(e)}")
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handler für Button-Klicks - ZENTRAL FÜR DEBUGGING
        """
        query = update.callback_query
        user = query.from_user
        chat_id = update.effective_chat.id
        callback_data = query.data
        message_id = query.message.message_id
        
        # SEHR AUSFÜHRLICHES LOGGING FÜR DEBUGGING
        self.logger.info(f"===== BUTTON GEKLICKT =====")
        self.logger.info(f"Button: {callback_data}")
        self.logger.info(f"User: {user.id} (@{user.username})")
        self.logger.info(f"Chat: {chat_id}")
        self.logger.info(f"Message ID: {message_id}")
        self.logger.info(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        try:
            # Bestätige den Button-Klick gegenüber Telegram
            self.logger.debug(f"Bestätige Button-Klick...")
            await query.answer()
            self.logger.debug(f"Button-Klick bestätigt")
            
            # Verarbeite je nach Button-Typ
            if callback_data == "portfolio":
                self.logger.debug(f"Verarbeite Portfolio-Button")
                await self._handle_portfolio(update, context)
            
            elif callback_data == "settings":
                self.logger.debug(f"Verarbeite Einstellungen-Button")
                await self._handle_settings(update, context)
            
            elif callback_data == "status":
                self.logger.debug(f"Verarbeite Status-Button")
                await self._handle_status(update, context)
            
            elif callback_data == "info":
                self.logger.debug(f"Verarbeite Info-Button")
                await self._handle_info(update, context)
            
            elif callback_data.startswith("back_"):
                self.logger.debug(f"Verarbeite Zurück-Button: {callback_data}")
                await self._handle_back(update, context, callback_data)
            
            else:
                self.logger.warning(f"Unbekannter Button: {callback_data}")
                await query.message.reply_text(f"Button nicht implementiert: {callback_data}")
            
            self.logger.info(f"===== BUTTON ERFOLGREICH VERARBEITET =====")
        
        except Exception as e:
            self.logger.error(f"FEHLER bei Button-Verarbeitung: {str(e)}")
            self.logger.error(f"Button-Daten: {callback_data}")
            self.logger.error(traceback.format_exc())
            
            try:
                await query.message.reply_text(
                    "⚠️ Es ist ein Fehler bei der Verarbeitung aufgetreten.\n"
                    "Details wurden im Log aufgezeichnet."
                )
            except:
                self.logger.error("Konnte keine Fehlermeldung senden")
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler für einfache Textnachrichten"""
        user = update.effective_user
        message_text = update.message.text
        
        self.logger.info(f"Nachricht empfangen | User: {user.id} | Text: '{message_text}'")
        
        try:
            await update.message.reply_text(
                "Ich verstehe nur Befehle und Buttons. Versuche /help für eine Liste der Befehle."
            )
        except Exception as e:
            self.logger.error(f"FEHLER beim Antworten auf Nachricht: {str(e)}")
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Fehlerbehandlung für den Bot"""
        self.logger.error(f"===== TELEGRAM-BOT FEHLER =====")
        
        # Fehlerdetails
        if context.error:
            self.logger.error(f"Error: {str(context.error)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update-Informationen
        if update:
            if update.effective_user:
                self.logger.error(f"User: {update.effective_user.id}")
            if update.effective_chat:
                self.logger.error(f"Chat: {update.effective_chat.id}")
            if update.effective_message:
                self.logger.error(f"Message: {update.effective_message.message_id}")
        
        # Admin benachrichtigen
        for admin_id in self.admin_ids:
            try:
                bot = context.bot
                await bot.send_message(
                    chat_id=admin_id,
                    text=f"⚠️ *Bot-Fehler aufgetreten*\n\n{str(context.error)[:200]}...",
                    parse_mode="Markdown"
                )
                self.logger.info(f"Admin {admin_id} über Fehler informiert")
            except:
                self.logger.error(f"Konnte Admin {admin_id} nicht informieren")
    
    # Button-Handler
    
    async def _handle_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Portfolio-Button"""
        query = update.callback_query
        self.logger.debug(f"Portfolio-Handler aufgerufen")
        
        keyboard = [
            [InlineKeyboardButton("🔙 Zurück", callback_data="back_main")]
        ]
        
        try:
            await query.message.edit_text(
                "📊 *Dein Portfolio:*\n\n"
                "• BTC: 0.05 BTC (1,500.00 EUR)\n"
                "• ETH: 0.5 ETH (750.00 EUR)\n"
                "• SOL: 10 SOL (500.00 EUR)\n\n"
                "Gesamtwert: 2,750.00 EUR\n"
                "24h Änderung: +3.5%",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.info(f"Portfolio-Ansicht angezeigt | User: {query.from_user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Anzeigen des Portfolios: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Einstellungen-Button"""
        query = update.callback_query
        self.logger.debug(f"Einstellungen-Handler aufgerufen")
        
        keyboard = [
            [
                InlineKeyboardButton("🔔 Benachrichtigungen", callback_data="settings_notif"),
                InlineKeyboardButton("💱 Währungen", callback_data="settings_currencies")
            ],
            [InlineKeyboardButton("🔙 Zurück", callback_data="back_main")]
        ]
        
        try:
            await query.message.edit_text(
                "⚙️ *Einstellungen:*\n\n"
                "Hier kannst du verschiedene Bot-Einstellungen anpassen.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.info(f"Einstellungen-Menü angezeigt | User: {query.from_user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Anzeigen der Einstellungen: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Status-Button"""
        query = update.callback_query
        self.logger.debug(f"Status-Handler aufgerufen")
        
        keyboard = [
            [InlineKeyboardButton("🔙 Zurück", callback_data="back_main")]
        ]
        
        try:
            await query.message.edit_text(
                "📈 *Bot-Status:*\n\n"
                "• Bot ist aktiv und läuft\n"
                "• Verbunden mit Telegram API\n"
                f"• Server-Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "• CPU-Auslastung: 25%\n"
                "• RAM-Nutzung: 512MB",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.info(f"Status-Ansicht angezeigt | User: {query.from_user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Anzeigen des Status: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Verarbeitet den Info-Button"""
        query = update.callback_query
        self.logger.debug(f"Info-Handler aufgerufen")
        
        keyboard = [
            [InlineKeyboardButton("🔙 Zurück", callback_data="back_main")]
        ]
        
        try:
            await query.message.edit_text(
                "🔍 *Bot-Info:*\n\n"
                "Trading Bot v1.0.0\n"
                "Entwickelt für automatisierten Kryptohandel\n\n"
                "*Funktionen:*\n"
                "• Automatischer Handel\n"
                "• Portfolio-Tracking\n"
                "• Marktanalyse\n",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            self.logger.info(f"Info-Ansicht angezeigt | User: {query.from_user.id}")
        except Exception as e:
            self.logger.error(f"FEHLER beim Anzeigen der Info: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str):
        """Verarbeitet Zurück-Buttons"""
        query = update.callback_query
        self.logger.debug(f"Zurück-Handler aufgerufen: {callback_data}")
        
        if callback_data == "back_main":
            # Zurück zum Hauptmenü
            keyboard = [
                [
                    InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                    InlineKeyboardButton("⚙️ Einstellungen", callback_data="settings")
                ],
                [
                    InlineKeyboardButton("📈 Status", callback_data="status"),
                    InlineKeyboardButton("🔍 Info", callback_data="info")
                ]
            ]
            
            try:
                await query.message.edit_text(
                    "Hauptmenü des Trading Bots. Wähle eine Option:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                self.logger.info(f"Zurück zum Hauptmenü | User: {query.from_user.id}")
            except Exception as e:
                self.logger.error(f"FEHLER beim Zurückkehren zum Hauptmenü: {str(e)}")
        else:
            self.logger.warning(f"Unbekannter Zurück-Button: {callback_data}")
    
    def run(self):
        """Startet den Bot im Polling-Modus"""
        self.logger.info("Starte Bot im Polling-Modus...")
        try:
            self.application.run_polling()
            self.logger.info("Bot beendet")
        except Exception as e:
            self.logger.critical(f"FEHLER beim Starten des Bots: {str(e)}")
            self.logger.critical(traceback.format_exc())
            raise

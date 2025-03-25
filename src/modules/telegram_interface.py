# telegram_interface.py

import os
import logging
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Union
from telegram import (
    Update, 
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    InputFile
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from src.core.config_manager import ConfigManager
from src.modules.main_controller import MainController

logger = logging.getLogger("TelegramInterface")

class TelegramInterface:
    """Core Telegram interface for trading bot control and monitoring"""
    
    def __init__(self, config: Dict[str, Any], main_controller: MainController):
        self.logger = logging.getLogger("TelegramInterface")
        self.config = config
        self.main_controller = main_controller
        self.allowed_users = self._load_allowed_users()
        self.application = None
        self.message_queue = []
        self._init_handlers()

    def _load_allowed_users(self) -> list:
        """Load allowed users from config"""
        users = self.config.get('allowed_users', '').split(',')
        return [int(u.strip()) for u in users if u.strip()]

    def start(self):
        """Start the Telegram bot"""
        self.logger.info("Starting Telegram interface...")
        token = self.config.get('bot_token')
        if not token:
            self.logger.error("No Telegram bot token found!")
            return

        self.application = Application.builder().token(token).build()
        
        # Register handlers
        self.application.add_handler(CommandHandler('start', self._start_handler))
        self.application.add_handler(CallbackQueryHandler(self._button_handler))
        self.application.add_handler(CommandHandler('logs', self._logs_handler))
        self.application.add_handler(MessageHandler(filters.TEXT, self._message_handler))

        # Set bot commands
        commands = [
            BotCommand("start", "Main control panel"),
            BotCommand("status", "Current bot status"),
            BotCommand("positions", "Open positions"),
            BotCommand("performance", "Performance metrics"),
            BotCommand("logs", "View recent logs"),
            BotCommand("stop", "Emergency shutdown")
        ]
        self.application.bot.set_my_commands(commands)
        
        # Start polling
        self.application.run_polling()
        self.logger.info("Telegram interface started")

    #region Core Handlers
    async def _start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not self._is_authorized(update):
            return
            
        keyboard = [
            [InlineKeyboardButton("🚀 Start Bot", callback_data="start_bot"),
             InlineKeyboardButton("🛑 Emergency Stop", callback_data="emergency_stop")],
            [InlineKeyboardButton("📊 Live Positions", callback_data="positions"),
             InlineKeyboardButton("📈 Performance", callback_data="performance")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("📋 Logs", callback_data="logs")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **Trading Bot Control Panel**\n\n"
            "Select an action:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses"""
        query = update.callback_query
        await query.answer()
        
        if not self._is_authorized(update):
            return

        handlers = {
            'start_bot': self._handle_start_bot,
            'emergency_stop': self._handle_emergency_stop,
            'positions': self._handle_positions,
            'performance': self._handle_performance,
            'settings': self._handle_settings,
            'logs': self._handle_logs
        }
        
        handler = handlers.get(query.data, self._handle_unknown)
        await handler(query)

    async def _handle_start_bot(self, query):
        """Handle bot startup"""
        success = self.main_controller.start()
        await self._send_operation_result(
            query, 
            "Bot startup initiated" if success else "Startup failed",
            success
        )
    #endregion

    #region Notification System
    async def send_notification(self, title: str, message: str, priority: str = "normal"):
        """Send formatted notification to all authorized users"""
        formatted_msg = f"🚨 **{title}**\n\n{message}"
        for user_id in self.allowed_users:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=formatted_msg,
                    parse_mode="Markdown"
                )
            except Exception as e:
                self.logger.error(f"Failed to send notification to {user_id}: {str(e)}")
    #endregion

    #region Utility Methods
    def _is_authorized(self, update: Update) -> bool:
        """Check if user is authorized"""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            self.logger.warning(f"Unauthorized access attempt from {user_id}")
            return False
        return True

    async def _send_operation_result(self, query, message: str, success: bool):
        """Send operation result with status icon"""
        icon = "✅" if success else "❌"
        await query.edit_message_text(
            text=f"{icon} {message}",
            reply_markup=None
        )
    #endregion

    #region Logging Integration
    async def _logs_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle logs command"""
        if not self._is_authorized(update):
            return
            
        try:
            with open("logs/trading_bot.log", "rb") as log_file:
                await update.message.reply_document(
                    document=InputFile(log_file),
                    caption="📄 Latest log file"
                )
        except Exception as e:
            await update.message.reply_text(f"❌ Error retrieving logs: {str(e)}")
    #endregion

    # Add other handler implementations following the same pattern...

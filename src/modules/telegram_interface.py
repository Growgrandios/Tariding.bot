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
from dotenv import load_dotenv

logger = logging.getLogger("TelegramInterface")

class TelegramInterface:
    """Core Telegram interface for trading bot control and monitoring"""
    
    def __init__(self, config: Dict[str, Any], main_controller):
        """
        Initialize Telegram interface
        
        Args:
            config: Configuration dictionary with Telegram settings
            main_controller: Reference to the MainController for bidirectional communication
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.config = config
        self.main_controller = main_controller
        self.allowed_users = self._load_allowed_users()
        self.application = None
        self.message_queue = []
        self.queue_thread = None
        
        # Initialize interface components
        self.logger.info("Initializing Telegram interface...")

    def _load_allowed_users(self) -> list:
        """Load allowed users from config"""
        users = self.config.get('allowed_users', '')
        if isinstance(users, str):
            return [int(u.strip()) for u in users.split(',') if u.strip()]
        elif isinstance(users, list):
            return [int(u) for u in users]
        return []

    def start(self):
        """Start the Telegram bot"""
        self.logger.info("Starting Telegram interface...")
        token = self.config.get('bot_token')
        if not token:
            self.logger.error("No Telegram bot token found!")
            return False

        try:
            # Initialize bot application
            self.application = Application.builder().token(token).build()
            
            # Register command handlers
            self.application.add_handler(CommandHandler('start', self._start_handler))
            self.application.add_handler(CommandHandler('status', self._status_handler))
            self.application.add_handler(CommandHandler('positions', self._positions_handler))
            self.application.add_handler(CommandHandler('performance', self._performance_handler))
            self.application.add_handler(CommandHandler('balance', self._balance_handler))
            self.application.add_handler(CommandHandler('logs', self._logs_handler))
            self.application.add_handler(CommandHandler('stop', self._stop_handler))
            self.application.add_handler(CommandHandler('help', self._help_handler))
            
            # Register callback query handler for buttons
            self.application.add_handler(CallbackQueryHandler(self._button_handler))
            
            # Register handler for text messages
            self.application.add_handler(MessageHandler(filters.TEXT, self._message_handler))

            # Set bot commands for menu
            commands = [
                BotCommand("start", "Main control panel"),
                BotCommand("status", "Current bot status"),
                BotCommand("positions", "View open positions"),
                BotCommand("performance", "Performance metrics"),
                BotCommand("balance", "Account balance"),
                BotCommand("logs", "View recent logs"),
                BotCommand("stop", "Emergency shutdown"),
                BotCommand("help", "Show help")
            ]
            self.application.bot.set_my_commands(commands)
            
            # Start message queue processor
            self.queue_thread = threading.Thread(target=self._process_message_queue, daemon=True)
            self.queue_thread.start()
            
            # Start polling
            self.application.run_polling()
            self.logger.info("Telegram interface started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Telegram interface: {str(e)}")
            return False

    async def _start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - entry point to the bot"""
        if not self._is_authorized(update):
            return
            
        keyboard = [
            [InlineKeyboardButton("🚀 Start Bot", callback_data="start_bot"),
             InlineKeyboardButton("🛑 Emergency Stop", callback_data="emergency_stop")],
            [InlineKeyboardButton("⏸️ Pause", callback_data="pause_bot"),
             InlineKeyboardButton("▶️ Resume", callback_data="resume_bot")],
            [InlineKeyboardButton("📊 Positions", callback_data="positions"),
             InlineKeyboardButton("📈 Performance", callback_data="performance")],
            [InlineKeyboardButton("💰 Account", callback_data="balance"),
             InlineKeyboardButton("📅 Reports", callback_data="reports")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("📋 Logs", callback_data="logs")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Get bot status
        status = self.main_controller.get_status()
        state = status.get('state', 'Unknown')
        state_emoji = self._get_state_emoji(state)
        
        await update.message.reply_text(
            f"🤖 *Trading Bot Control Panel*\n\n"
            f"Status: {state_emoji} {state}\n"
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"Select an action from the menu below:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    def _get_state_emoji(self, state: str) -> str:
        """Get emoji representation of bot state"""
        return {
            "initializing": "🔄",
            "ready": "✅",
            "running": "🚀",
            "paused": "⏸️",
            "stopping": "🛑",
            "error": "❌",
            "maintenance": "🔧",
            "emergency": "🚨"
        }.get(state.lower(), "❓")

    async def _status_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show detailed bot status"""
        if not self._is_authorized(update):
            return
        
        status = self.main_controller.get_status()
        
        # Format detailed status message
        message = (
            f"🤖 *Bot Status*\n\n"
            f"State: {self._get_state_emoji(status.get('state', 'Unknown'))} {status.get('state', 'Unknown')}\n"
            f"Emergency Mode: {'🚨 Enabled' if status.get('emergency_mode', False) else '✅ Disabled'}\n"
            f"Running: {'✅ Yes' if status.get('running', False) else '❌ No'}\n\n"
            f"*Module Status:*\n"
        )
        
        # Add module statuses
        modules = status.get('modules', {})
        for module_name, module_status in modules.items():
            status_text = module_status.get('status', 'unknown')
            status_emoji = "✅" if status_text in ["running", "initialized"] else "❌" if status_text in ["error", "stopped"] else "⚠️"
            message += f"• {status_emoji} {module_name}: {status_text}\n"
        
        # Add recent events if available
        events = status.get('events', [])
        if events:
            message += "\n*Recent Events:*\n"
            for event in events[-3:]:  # Show last 3 events
                event_time = datetime.fromisoformat(event.get('timestamp', '')).strftime('%H:%M:%S')
                message += f"• {event_time} - {event.get('title', 'Unknown event')}\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _positions_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - show open positions"""
        if not self._is_authorized(update):
            return
        
        await self._handle_positions(update)

    async def _performance_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command - show performance metrics"""
        if not self._is_authorized(update):
            return
        
        await self._handle_performance(update)

    async def _balance_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command - show account balance"""
        if not self._is_authorized(update):
            return
        
        balance = self.main_controller._get_account_balance()
        
        if balance.get('status') != 'success':
            await update.message.reply_text(f"❌ Failed to retrieve balance: {balance.get('message', 'Unknown error')}")
            return
        
        # Format balance information
        balance_data = balance.get('balance', {})
        message = "💰 *Account Balance*\n\n"
        
        # Total balance
        total_balance = balance_data.get('total_balance_usd', 0)
        message += f"Total Balance: ${total_balance:,.2f}\n\n"
        
        # Available and allocated balance
        available = balance_data.get('available_balance_usd', 0)
        allocated = balance_data.get('allocated_balance_usd', 0)
        message += f"Available: ${available:,.2f}\n"
        message += f"Allocated: ${allocated:,.2f}\n\n"
        
        # Assets breakdown if available
        assets = balance_data.get('assets', [])
        if assets:
            message += "*Assets:*\n"
            for asset in assets:
                symbol = asset.get('symbol', '???')
                amount = asset.get('amount', 0)
                value_usd = asset.get('value_usd', 0)
                message += f"• {symbol}: {amount:,.8f} (${value_usd:,.2f})\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="balance")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _logs_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command - show and manage logs"""
        if not self._is_authorized(update):
            return
            
        await self._handle_logs(update)

    async def _stop_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - emergency bot shutdown"""
        if not self._is_authorized(update):
            return
        
        # Confirmation keyboard
        keyboard = [
            [InlineKeyboardButton("✅ Yes, stop the bot", callback_data="confirm_emergency_stop")],
            [InlineKeyboardButton("❌ No, cancel", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚨 *EMERGENCY SHUTDOWN*\n\n"
            "Are you sure you want to stop the bot?\n"
            "This will close all positions and halt all trading activities.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command - show available commands"""
        if not self._is_authorized(update):
            return
        
        message = (
            "🤖 *Trading Bot Help*\n\n"
            "*Trading Controls:*\n"
            "• `/start` - Main control panel\n"
            "• `/status` - Current bot status\n"
            "• `/stop` - Emergency shutdown\n\n"
            
            "*Trading Info:*\n"
            "• `/positions` - View open positions\n"
            "• `/balance` - Account balance\n"
            "• `/performance` - Performance metrics\n\n"
            
            "*System:*\n"
            "• `/logs` - View recent logs\n"
            "• `/help` - Show this help message\n\n"
            
            "*Tips:*\n"
            "• Use /start to access the main control panel\n"
            "• Daily reports are generated automatically at midnight\n"
            "• Tax calculations are available in the performance section"
        )
        
        keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callback queries"""
        query = update.callback_query
        await query.answer()  # Answer to stop loading animation
        
        if not self._is_authorized(update):
            return
        
        callback_data = query.data
        
        # Handle confirmation callbacks
        if callback_data == "confirm_emergency_stop":
            await self._handle_emergency_stop(query)
            return
            
        # Handle general callbacks
        handler_map = {
            "start_bot": self._handle_start_bot,
            "emergency_stop": self._confirm_emergency_stop,
            "pause_bot": self._handle_pause_bot,
            "resume_bot": self._handle_resume_bot,
            "positions": self._handle_positions,
            "performance": self._handle_performance,
            "balance": self._handle_balance,
            "reports": self._handle_reports,
            "settings": self._handle_settings,
            "logs": self._handle_logs,
            "refresh_status": self._handle_refresh_status,
            "back_to_main": self._handle_back_to_main,
            "tax_calculator": self._handle_tax_calculator,
            "daily_report": self._handle_daily_report,
            "weekly_report": self._handle_weekly_report,
            "debug_mode": self._handle_debug_mode,
            "remote_start": self._handle_remote_start
        }
        
        handler = handler_map.get(callback_data)
        if handler:
            await handler(query)
        else:
            self.logger.warning(f"Unhandled callback data: {callback_data}")
            await query.edit_message_text(
                "⚠️ Unrecognized action.\n\nUse /start to return to the main menu."
            )

    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        if not self._is_authorized(update):
            return
        
        # Simple text message handler - provide guidance
        await update.message.reply_text(
            "I'm here to help you control your trading bot. "
            "Use /start to access the main control panel, or /help to see available commands."
        )

    async def _handle_start_bot(self, query):
        """Handle start_bot callback - start the trading bot"""
        await query.edit_message_text("Starting bot, please wait...")
        
        result = self.main_controller.start()
        
        if result:
            message = "✅ Bot started successfully!\n\nTrading is now active."
        else:
            message = "❌ Failed to start the bot.\n\nCheck logs for details."
        
        keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup
        )

    async def _confirm_emergency_stop(self, query):
        """Handle emergency_stop callback - confirm emergency shutdown"""
        keyboard = [
            [InlineKeyboardButton("✅ Yes, stop the bot", callback_data="confirm_emergency_stop")],
            [InlineKeyboardButton("❌ No, cancel", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🚨 *EMERGENCY SHUTDOWN*\n\n"
            "Are you sure you want to stop the bot?\n"
            "This will close all positions and halt all trading activities.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _handle_emergency_stop(self, query):
        """Execute emergency shutdown after confirmation"""
        await query.edit_message_text("🚨 EMERGENCY SHUTDOWN IN PROGRESS...")
        
        result = self.main_controller.stop()
        
        if result:
            message = "✅ Bot emergency shutdown completed.\n\nAll positions have been closed and trading has been halted."
        else:
            message = "⚠️ Emergency shutdown initiated but encountered issues.\n\nPlease check logs for details."
        
        keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup
        )

    async def _handle_pause_bot(self, query):
        """Handle pause_bot callback - pause the trading bot"""
        await query.edit_message_text("Pausing bot, please wait...")
        
        result = self.main_controller.pause()
        
        if result:
            message = "⏸️ Bot paused successfully.\n\nNo new trades will be opened, but existing positions remain active."
        else:
            message = "❌ Failed to pause the bot.\n\nCheck logs for details."
        
        keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup
        )

    async def _handle_resume_bot(self, query):
        """Handle resume_bot callback - resume the trading bot"""
        await query.edit_message_text("Resuming bot, please wait...")
        
        result = self.main_controller.resume()
        
        if result:
            message = "▶️ Bot resumed successfully.\n\nTrading activity has been restored."
        else:
            message = "❌ Failed to resume the bot.\n\nCheck logs for details."
        
        keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup
        )

    async def _handle_positions(self, update):
        """Handle positions callback - show open positions"""
        # Determine if update is from message or callback query
        is_callback = hasattr(update, 'callback_query')
        query = update.callback_query if is_callback else None
        
        if is_callback:
            await query.edit_message_text("Loading positions, please wait...")
        
        positions_data = self.main_controller._get_open_positions()
        
        if positions_data.get('status') != 'success':
            message = f"❌ Failed to retrieve positions: {positions_data.get('message', 'Unknown error')}"
            
            if is_callback:
                await query.edit_message_text(message)
            else:
                await update.message.reply_text(message)
            return
        
        positions = positions_data.get('positions', [])
        
        if not positions:
            message = "📊 *Open Positions*\n\nNo open positions at the moment."
        else:
            message = "📊 *Open Positions*\n\n"
            
            for position in positions:
                symbol = position.get('symbol', '???')
                side = position.get('side', '???').upper()
                entry_price = position.get('entry_price', 0)
                current_price = position.get('current_price', 0)
                pnl = position.get('pnl', 0)
                pnl_percent = position.get('pnl_percent', 0)
                size = position.get('size', 0)
                leverage = position.get('leverage', 1)
                
                # Format with emojis based on position type and profitability
                side_emoji = "🟢" if side == "LONG" else "🔴"
                pnl_emoji = "📈" if pnl >= 0 else "📉"
                
                message += (
                    f"{side_emoji} *{symbol}* ({side} x{leverage})\n"
                    f"Size: {size}\n"
                    f"Entry: ${entry_price:,.2f} → Current: ${current_price:,.2f}\n"
                    f"PnL: {pnl_emoji} ${pnl:,.2f} ({pnl_percent:,.2f}%)\n\n"
                )
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="positions")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def _handle_performance(self, update):
        """Handle performance callback - show performance metrics"""
        # Determine if update is from message or callback query
        is_callback = hasattr(update, 'callback_query')
        query = update.callback_query if is_callback else None
        
        if is_callback:
            await query.edit_message_text("Loading performance data, please wait...")
        
        performance_data = self.main_controller._get_performance_metrics()
        
        if performance_data.get('status') != 'success':
            message = f"❌ Failed to retrieve performance data: {performance_data.get('message', 'Unknown error')}"
            
            if is_callback:
                await query.edit_message_text(message)
            else:
                await update.message.reply_text(message)
            return
        
        metrics = performance_data.get('metrics', {})
        
        # Build the performance message
        message = "📈 *Performance Metrics*\n\n"
        
        # Trading metrics
        trading_metrics = metrics.get('trading', {})
        if trading_metrics:
            total_trades = trading_metrics.get('total_trades', 0)
            win_rate = trading_metrics.get('win_rate', 0) * 100
            avg_win = trading_metrics.get('avg_win', 0)
            avg_loss = trading_metrics.get('avg_loss', 0)
            total_pnl = trading_metrics.get('total_pnl', 0)
            
            message += (
                f"*Trading Performance:*\n"
                f"Total Trades: {total_trades}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Avg Win: {avg_win:.2f}%\n"
                f"Avg Loss: {avg_loss:.2f}%\n"
                f"Total PnL: {total_pnl:.2f}%\n\n"
            )
        
        # Tax information
        tax_metrics = metrics.get('tax', {})
        if tax_metrics:
            realized_profit = tax_metrics.get('realized_profit', 0)
            tax_reserve = tax_metrics.get('tax_reserve', 0)
            
            message += (
                f"*Tax Information:*\n"
                f"Realized Profit: ${realized_profit:,.2f}\n"
                f"Tax Reserve: ${tax_reserve:,.2f}\n"
            )
        
        keyboard = [
            [
                InlineKeyboardButton("📅 Daily Report", callback_data="daily_report"),
                InlineKeyboardButton("💰 Tax Info", callback_data="tax_calculator")
            ],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if is_callback:
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def _handle_balance(self, query):
        """Handle balance callback - show account balance"""
        await query.edit_message_text("Loading account balance, please wait...")
        
        balance = self.main_controller._get_account_balance()
        
        if balance.get('status') != 'success':
            await query.edit_message_text(f"❌ Failed to retrieve account balance: {balance.get('message', 'Unknown error')}")
            return
        
        # Format balance information
        balance_data = balance.get('balance', {})
        message = "💰 *Account Balance*\n\n"
        
        # Total balance
        total_balance = balance_data.get('total_balance_usd', 0)
        message += f"Total Balance: ${total_balance:,.2f}\n\n"
        
        # Available and allocated balance
        available = balance_data.get('available_balance_usd', 0)
        allocated = balance_data.get('allocated_balance_usd', 0)
        message += f"Available: ${available:,.2f}\n"
        message += f"Allocated: ${allocated:,.2f}\n\n"
        
        # Assets breakdown if available
        assets = balance_data.get('assets', [])
        if assets:
            message += "*Assets:*\n"
            for asset in assets:
                symbol = asset.get('symbol', '???')
                amount = asset.get('amount', 0)
                value_usd = asset.get('value_usd', 0)
                message += f"• {symbol}: {amount:,.8f} (${value_usd:,.2f})\n"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="balance")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _handle_reports(self, query):
        """Handle reports callback - show report options"""
        keyboard = [
            [InlineKeyboardButton("📊 Daily Report", callback_data="daily_report")],
            [InlineKeyboardButton("📈 Weekly Report", callback_data="weekly_report")],
            [InlineKeyboardButton("💰 Tax Calculator", callback_data="tax_calculator")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📋 *Reports*\n\n"
            "Select a report to generate:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _handle_settings(self, query):
        """Handle settings callback - show and modify bot settings"""
        # Get current configuration
        general_config = self.main_controller.config_manager.get_config('general')
        trading_config = self.main_controller.config_manager.get_config('trading')
        
        message = (
            "⚙️ *Bot Settings*\n\n"
            f"*General:*\n"
            f"• Bot Name: {general_config.get('bot_name', 'Unknown')}\n"
            f"• Log Level: {general_config.get('log_level', 'INFO')}\n\n"
            
            f"*Trading:*\n"
            f"• Mode: {trading_config.get('mode', 'paper')}\n"
            f"• Max Leverage: {trading_config.get('max_leverage', 'Unknown')}\n"
            f"• Risk Per Trade: {trading_config.get('risk_per_trade', 0) * 100:.1f}%\n"
            f"• Max Open Trades: {trading_config.get('max_open_trades', 'Unknown')}\n"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔧 Debug Mode", callback_data="debug_mode")],
            [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _handle_logs(self, update):
        """Handle logs callback - show recent logs"""
        # Determine if update is from message or callback query
        is_callback = hasattr(update, 'callback_query')
        query = update.callback_query if is_callback else None
        
        if is_callback:
            await query.edit_message_text("Loading logs, please wait...")
        
        try:
            # Read recent log entries
            log_path = "logs/trading_bot.log"
            
            try:
                with open(log_path, 'r') as log_file:
                    log_lines = log_file.readlines()
                    # Get the last 20 lines
                    recent_logs = log_lines[-20:]
            except FileNotFoundError:
                recent_logs = ["Log file not found."]
            except Exception as e:
                recent_logs = [f"Error reading log file: {str(e)}"]
            
            # Format the logs
            log_text = "".join(recent_logs)
            
            # If log is too long, truncate it
            if len(log_text) > 3800:
                log_text = log_text[-3800:]
                log_text = "...[truncated]...\n" + log_text
            
            # Format the message
            message = f"📋 *Recent Logs*\n\n``````"
            
            # Send log file if message is too long for Telegram
            if len(message) > 4096:
                # Send the file instead
                with open(log_path, 'rb') as log_file:
                    if is_callback:
                        await query.edit_message_text("Log too large for display, sending as file...")
                        await query.message.reply_document(
                            document=InputFile(log_file, filename="trading_bot_logs.txt"),
                            caption="📋 Recent Logs"
                        )
                    else:
                        await update.message.reply_document(
                            document=InputFile(log_file, filename="trading_bot_logs.txt"),
                            caption="📋 Recent Logs"
                        )
                
                # Update original message if this was a callback
                if is_callback:
                    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.edit_message_text(
                        "Logs have been sent as a separate file due to size limitations.",
                        reply_markup=reply_markup
                    )
                return
            
            # Add buttons for log options
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="logs"),
                    InlineKeyboardButton("📥 Download Full Logs", callback_data="download_logs")
                ],
                [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if is_callback:
                await query.edit_message_text(
                    message,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    message,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            
        except Exception as e:
            self.logger.error(f"Error displaying logs: {str(e)}")
            error_message = f"❌ Error displaying logs: {str(e)}"
            
            if is_callback:
                await query.edit_message_text(error_message)
            else:
                await update.message.reply_text(error_message)

    async def _handle_refresh_status(self, query):
        """Handle refresh_status callback - refresh the status information"""
        # Just reuse the status handler for the query's message
        await self._status_handler(query.message, None)

    async def _handle_back_to_main(self, query):
        """Handle back_to_main callback - return to main menu"""
        # Reuse the start handler for the query's message
        await self._start_handler(query.message, None)

    async def _handle_tax_calculator(self, query):
        """Handle tax_calculator callback - show tax calculations"""
        await query.edit_message_text("Calculating tax information, please wait...")
        
        try:
            # Get tax data from performance metrics
            performance_data = self.main_controller._get_performance_metrics()
            
            if performance_data.get('status') != 'success' or 'tax' not in performance_data.get('metrics', {}):
                await query.edit_message_text(
                    "❌ Failed to retrieve tax information. Tax module may not be properly configured."
                )
                return
            
            tax_data = performance_data['metrics']['tax']
            
            # Format the tax report
            tax_report = (
                f"💰 *Tax Calculation Report*\n\n"
                
                f"*Realized P&L:*\n"
                f"Total Profit: ${tax_data.get('realized_profit', 0):,.2f}\n"
                f"Total Loss: ${tax_data.get('realized_loss', 0):,.2f}\n"
                f"Net P&L: ${tax_data.get('net_pnl', 0):,.2f}\n\n"
                
                f"*Tax Estimates:*\n"
                f"Tax Method: {tax_data.get('tax_method', 'Unknown')}\n"
                f"Estimated Tax: ${tax_data.get('tax_reserve', 0):,.2f}\n\n"
                
                f"*Trading Period:*\n"
                f"Start Date: {tax_data.get('period_start', 'Unknown')}\n"
                f"End Date: {tax_data.get('period_end', datetime.now().strftime('%Y-%m-%d'))}\n\n"
                
                f"*Disclaimer:*\n"
                f"This is an estimate only. Please consult a tax professional for accurate tax advice."
            )
            
            keyboard = [
                [InlineKeyboardButton("💾 Export Tax Report", callback_data="export_tax_report")],
                [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="back_to_main")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                tax_report,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            self.logger.error(f"Error generating tax report: {str(e)}")
            await query.edit_message_text(
                f"❌ Error generating tax report: {str(e)}\n\nCheck logs for details."
            )

    async def _handle_daily_report(self, query):
        """Handle daily_report callback - show daily performance report"""
        await query.edit_message_text("Generating daily report, please wait...")
        
        try:
            # Get performance data
            performance_data = self.main_controller._get_performance_metrics()
            positions_data = self.main_controller._get_open_positions()
            balance_data = self.main_controller._get_account_balance()
            
            # Check if all data was retrieved successfully
            if any(data.get('status') != 'success' for data in [performance_data, positions_data, balance_data]):
                await query.edit_message_text(
                    "❌ Failed to generate daily report: Unable to retrieve all required data."
                )
                return
            
            # Extract metrics
            metrics = performance_data.get('metri

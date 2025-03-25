# telegram_interface.py

import os
import logging
import threading
import time
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import traceback
import subprocess

# Telegram libraries
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

class TelegramInterface:
    """
    Telegram interface for the trading bot.
    Provides a user-friendly way to control the bot, receive notifications,
    and view performance data.
    """

    def __init__(self, config: Dict[str, Any], main_controller=None):
        """
        Initializes the TelegramInterface.
        
        Args:
            config: Configuration dictionary with Telegram-specific settings
            main_controller: Reference to the MainController for callback functions
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initializing TelegramInterface...")
        
        # Save configuration
        self.config = config or {}
        
        # Reference to the main controller
        self.main_controller = main_controller
        
        # Parse configuration
        self.bot_token = config.get('bot_token', '')
        self.allowed_users = self._parse_allowed_users(config.get('allowed_users', ''))
        self.notification_level = config.get('notification_level', 'INFO').upper()
        self.enabled = config.get('enabled', True)
        self.commands_enabled = config.get('commands_enabled', True)
        self.status_update_interval = config.get('status_update_interval', 3600)  # Default: hourly
        
        # Command callbacks dictionary
        self.command_handlers = {}
        
        # Priority emojis for notifications
        self.priority_emoji = {
            "low": "ℹ️",
            "normal": "📊", 
            "high": "⚠️",
            "critical": "🚨"
        }
        
        # Setup bot if enabled
        if self.enabled and self.bot_token:
            try:
                # Initialize bot
                self.updater = Updater(self.bot_token)
                self.bot = self.updater.bot
                self.dp = self.updater.dispatcher
                
                # Register handlers
                self._register_handlers()
                
                self.logger.info("Telegram bot initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing Telegram bot: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.enabled = False
        else:
            self.logger.warning("Telegram bot disabled or no token provided")
            self.enabled = False
        
        # Periodic update thread
        self.update_thread = None
        self.running = False
        
        # Chart and visualization settings
        self.chart_settings = {
            'default_figsize': (10, 6),
            'style': 'dark_background',
            'dpi': 100
        }
        
        self.logger.info("TelegramInterface initialization complete")
    
    def _parse_allowed_users(self, allowed_users_str: str) -> List[int]:
        """
        Parses the allowed users string into a list of user IDs.
        
        Args:
            allowed_users_str: Comma-separated string of user IDs
            
        Returns:
            List of integer user IDs
        """
        try:
            if not allowed_users_str:
                return []
            
            # Split by comma and convert to integers
            return [int(user_id.strip()) for user_id in allowed_users_str.split(',') if user_id.strip()]
        except Exception as e:
            self.logger.error(f"Error parsing allowed users: {str(e)}")
            return []
    
    def _register_handlers(self):
        """Registers all command and message handlers."""
        # Main command handlers
        self.dp.add_handler(CommandHandler("start", self._cmd_start))
        self.dp.add_handler(CommandHandler("help", self._cmd_help))
        self.dp.add_handler(CommandHandler("status", self._cmd_status))
        self.dp.add_handler(CommandHandler("menu", self._cmd_menu))
        self.dp.add_handler(CommandHandler("balance", self._cmd_balance))
        self.dp.add_handler(CommandHandler("positions", self._cmd_positions))
        self.dp.add_handler(CommandHandler("start_bot", self._cmd_start_bot))
        self.dp.add_handler(CommandHandler("stop_bot", self._cmd_stop_bot))
        self.dp.add_handler(CommandHandler("daily_report", self._cmd_daily_report))
        self.dp.add_handler(CommandHandler("tax", self._cmd_tax_summary))
        
        # Callback query handler for button clicks
        self.dp.add_handler(CallbackQueryHandler(self._button_callback))
        
        # Error handler
        self.dp.add_error_handler(self._error_handler)
    
    def start(self):
        """Starts the Telegram bot."""
        if not self.enabled:
            self.logger.warning("Telegram bot is disabled, not starting")
            return False
        
        try:
            # Start the bot polling
            self.updater.start_polling()
            
            # Start periodic update thread if configured
            if self.status_update_interval > 0:
                self.running = True
                self.update_thread = threading.Thread(target=self._periodic_update_loop, daemon=True)
                self.update_thread.start()
            
            self.logger.info("Telegram bot started successfully")
            
            # Notify admins
            self._notify_admins("Bot Started", "Trading bot has been started and is now operational.")
            
            return True
        except Exception as e:
            self.logger.error(f"Error starting Telegram bot: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def stop(self):
        """Stops the Telegram bot."""
        if not self.enabled:
            return
        
        try:
            # Stop the periodic update thread
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            # Stop the bot
            self.updater.stop()
            
            self.logger.info("Telegram bot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping Telegram bot: {str(e)}")
    
    def _periodic_update_loop(self):
        """Periodic updates loop for sending regular status updates."""
        self.logger.info(f"Starting periodic update loop (interval: {self.status_update_interval}s)")
        
        last_daily_report_day = -1
        last_weekly_report_day = -1
        
        while self.running:
            try:
                now = datetime.datetime.now()
                
                # Daily report at 8:00 AM
                if now.hour == 8 and now.minute < 5 and now.day != last_daily_report_day:
                    self.send_daily_report()
                    last_daily_report_day = now.day
                
                # Weekly report on Monday at 8:00 AM
                if now.weekday() == 0 and now.hour == 8 and now.minute < 5 and now.day != last_weekly_report_day:
                    self.send_weekly_report()
                    last_weekly_report_day = now.day
                
                # Sleep for 60 seconds before checking again
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in periodic update loop: {str(e)}")
                time.sleep(300)  # Sleep longer after error
    
    def register_commands(self, commands: Dict[str, Callable]):
        """
        Registers external command callbacks.
        
        Args:
            commands: Dictionary mapping command names to callback functions
        """
        if not self.enabled:
            return
        
        self.command_handlers.update(commands)
        self.logger.info(f"Registered {len(commands)} external commands")
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sends a notification to all allowed users.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level ("low", "normal", "high", "critical")
        """
        if not self.enabled:
            return
        
        # Skip low priority notifications if level is not DEBUG
        if priority == "low" and self.notification_level not in ["DEBUG", "TRACE"]:
            return
        
        # Format based on priority
        priority_emoji = {
            "low": "ℹ️",
            "normal": "

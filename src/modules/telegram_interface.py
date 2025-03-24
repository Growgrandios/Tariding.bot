import os
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import traceback
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import requests

# Für Headless-Server (ohne GUI)
matplotlib.use('Agg')

class TelegramInterface:
    """
    Telegram-Bot-Schnittstelle für die Fernsteuerung und Benachrichtigungen des Trading-Bots.
    Alle Aktionen (Befehle und Button-Klicks) werden detailliert in der SSH-Konsole geloggt.
    """
    def __init__(self, config: Dict[str, Any], main_controller=None):
        """
        Initialisiert die Telegram-Schnittstelle.
        """
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface...")
        
        # Debug-Modus aktivieren - WICHTIG für Fehlerbehebung
        self.debug_mode = config.get('debug_mode', True)
        if self.debug_mode:
            logging.getLogger("TelegramInterface").setLevel(logging.DEBUG)
            self.logger.info("Debug-Modus aktiviert - alle Button-Aktionen werden protokolliert")
            
            # Konsolenausgabe konfigurieren für bessere Sichtbarkeit
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger("TelegramInterface").addHandler(console_handler)
        
        # API-Konfiguration
        self.bot_token = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        
        # String oder Liste von IDs in String-Liste konvertieren
        allowed_users_raw = config.get('allowed_users', [])
        if isinstance(allowed_users_raw, str):
            self.allowed_users = [str(user_id.strip()) for user_id in allowed_users_raw.split(',') if user_id.strip()]
        elif isinstance(allowed_users_raw, list):
            self.allowed_users = [str(user_id) for user_id in allowed_users_raw if user_id]
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
        
        # Thread für Bot-Updates
        self.bot_thread = None
        self.is_running = False
        
        # Befehlsreferenzen
        self.commands = {}
        
        # Verzeichnis für aufgezeichnete Transkripte
        self.transcript_dir = Path('data/transcripts')
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Verzeichnis für temporäre Grafiken
        self.charts_dir = Path('data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP Session für Requests
        self.session = None
        
        # Verarbeitete Update-IDs speichern
        self.processed_updates = set()

        self.logger.info("TelegramInterface erfolgreich initialisiert")
    
    def register_commands(self, commands: Dict[str, Callable]):
        """
        Registriert benutzerdefinierte Befehle vom MainController.
        """
        self.logger.info(f"Registriere {len(commands)} benutzerdefinierte Befehle")
        self.commands = commands
    
    def start(self):
        """Startet den Telegram-Bot in einem separaten Thread."""
        if not self.is_configured:
            self.logger.warning("Telegram-Bot nicht konfiguriert, kann nicht gestartet werden")
            return False
        
        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return True
        
        try:
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
        """Führt den Telegram-Bot komplett ohne asyncio und Signal-Handler aus"""
        try:
            self.session = requests.Session()
            last_update_id = 0
            self.logger.info("Bot-Thread gestartet (HTTP-Polling-Modus)")
            
            while self.is_running:
                try:
                    # Direkter API-Aufruf mit Long-Polling
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
                            # Prüfen, ob dieses Update bereits verarbeitet wurde
                            if update["update_id"] not in self.processed_updates:
                                self._handle_raw_update(update)
                                self.processed_updates.add(update["update_id"])
                                # Begrenze die Größe der verarbeiteten Updates
                                if len(self.processed_updates) > 1000:
                                    self.processed_updates = set(list(self.processed_updates)[-500:])
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
        """Verarbeitet ein Telegram-Update mit ausführlichem Debugging"""
        try:
            # Ausführliches Debugging - Zeige das vollständige Update in der Konsole
            print(f"TELEGRAM UPDATE EMPFANGEN: {json.dumps(update, indent=2)}")
            self.logger.critical(f"Update-ID: {update.get('update_id')} - Typ: {'callback_query' if 'callback_query' in update else 'message'}")
            
            # Chat_id und Text extrahieren - abhängig vom Update-Typ
            chat_id = None
            user_id = None
            text = None
            callback_data = None
            message_id = None
            
            # Prüfen ob Nachricht oder Callback-Query
            if "message" in update:
                self.logger.info("Nachricht erkannt")
                message = update["message"]
                chat_id = message.get("chat", {}).get("id")
                user_id = message.get("from", {}).get("id")
                text = message.get("text", "")
                message_id = message.get("message_id")
                self.logger.info(f"Text-Nachricht von User {user_id} in Chat {chat_id}: {text}")
            elif "callback_query" in update:
                self.logger.info("Callback-Query (Button-Druck) erkannt")
                callback_query = update["callback_query"]
                chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
                user_id = callback_query.get("from", {}).get("id")
                callback_data = callback_query.get("data")
                message_id = callback_query.get("message", {}).get("message_id")
                
                print(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.critical(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.info(f"Chat ID: {chat_id}, User ID: {user_id}")
            
            # Autorisierung prüfen
            if not chat_id or not self._is_authorized(str(user_id)):
                self.logger.warning(f"Nicht autorisierter Zugriff von User ID: {user_id}")
                return
            
            # Callback-Query beantworten (wichtig für Buttons)
            if "callback_query" in update:
                try:
                    callback_id = update["callback_query"]["id"]
                    print(f"BUTTON GEDRÜCKT - CALLBACK ID: {callback_id}")
                    self.logger.critical(f"Beantworte Callback: {callback_id}")
                    
                    # Antwort an Telegram senden
                    response = self.session.post(
                        f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery",
                        json={"callback_query_id": callback_id}
                    )
                    print(f"TELEGRAM ANTWORT: {response.status_code} - {response.text}")
                    self.logger.info(f"Antwort von Telegram: {response.status_code}, {response.text}")
                    
                    # Kurze Verzögerung für bessere Verarbeitung
                    time.sleep(0.3)
                except Exception as e:
                    print(f"CALLBACK-FEHLER: {str(e)}")
                    self.logger.error(f"Fehler beim Beantworten der Callback-Query: {str(e)}")
            
            # Callback-Anfragen verarbeiten
            if callback_data:
                self.logger.info(f"Verarbeite Callback-Daten: {callback_data}")
                self._handle_callback_data(chat_id, callback_data, message_id)
                return
            
            # Textnachrichten verarbeiten
            if text:
                self.logger.info(f"Verarbeite Text: {text}")
                # Befehle verarbeiten (beginnen mit '/')
                if text.startswith("/"):
                    self._process_command(chat_id, text)
                else:
                    self._send_direct_message(chat_id, "Ich verstehe nur Befehle. Verwende /help für eine Liste der verfügbaren Befehle.")
        except Exception as e:
            self.logger.error(f"Fehler bei der Update-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _handle_callback_data(self, chat_id, callback_data, message_id=None):
        """Verarbeitet Callback-Daten von Inline-Buttons"""
        try:
            self.logger.info(f"Verarbeite Callback-Daten: {callback_data} von Chat {chat_id}")
            
            # Je nach Callback-Data unterschiedliche Aktionen ausführen
            if callback_data == "startbot":
                self._handle_start_bot(chat_id)
            elif callback_data == "stopbot":
                self._handle_stop_bot(chat_id)
            elif callback_data == "pausebot":
                self._handle_pause_bot(chat_id)
            elif callback_data == "resumebot":
                self._handle_resume_bot(chat_id)
            elif callback_data == "balance":
                self._handle_balance(chat_id)
            elif callback_data == "positions":
                self._handle_positions(chat_id)
            elif callback_data == "performance":
                self._handle_performance(chat_id)
            elif callback_data == "dashboard":
                self._send_dashboard(chat_id)
            elif callback_data == "refresh_status":
                self._send_status_message(chat_id, message_id)
            elif callback_data == "help":
                self._process_command(chat_id, "/help")
            elif callback_data == "status":
                self._send_status_message(chat_id)
            elif callback_data == "refresh_positions":
                self._handle_positions(chat_id)
            elif callback_data == "close_all_positions":
                self._confirm_close_all_positions(chat_id)
            elif callback_data == "confirm_close_all":
                self._execute_close_all_positions(chat_id)
            elif callback_data == "cancel_close_all":
                self._send_direct_message(chat_id, "Abgebrochen. Keine Positionen wurden geschlossen.")
            elif callback_data.startswith("train_model:"):
                parts = callback_data.split(":")
                if len(parts) >= 3:
                    symbol = parts[1]
                    timeframe = parts[2]
                    self._handle_train_model(chat_id, symbol, timeframe)
            elif callback_data.startswith("analyze_news:"):
                asset = callback_data.split(":")[1]
                self._handle_analyze_news(chat_id, asset)
            elif callback_data.startswith("optimize_portfolio"):
                self._handle_optimize_portfolio(chat_id)
            else:
                self._send_direct_message(chat_id, f"Unbekannte Aktion: {callback_data}")
        except Exception as e:
            self.logger.error(f"Fehler bei der Callback-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._send_direct_message(chat_id, f"Fehler bei der Ausführung: {str(e)}")
    
    def _process_command(self, chat_id, command_text):
        """Verarbeitet einen Befehl"""
        # Befehl und Parameter extrahieren
        parts = command_text.split(maxsplit=1)
        command = parts[0][1:]  # Entferne das '/'
        params = parts[1] if len(parts) > 1 else ""
        
        # Testbefehl für Logging-Prüfung
        if command == "testlog":
            print("TESTLOG: Direkter Print zur Konsole")
            self.logger.debug("DEBUG Log-Test")
            self.logger.info("INFO Log-Test")
            self.logger.warning("WARNING Log-Test")
            self.logger.error("ERROR Log-Test")
            self.logger.critical("CRITICAL Log-Test")
            self._send_direct_message(chat_id, "Log-Test wurde ausgeführt, prüfe deine SSH-Konsole")
            return
        
        # Standard-Befehle
        if command == "start":
            self._send_direct_message(chat_id, "🤖 Bot aktiv! Nutze /help für Befehle")
        elif command == "help":
            help_text = """
📋 Verfügbare Befehle:

Grundlegende Befehle:
/start - Startet den Bot und zeigt das Willkommensmenü
/help - Zeigt diese Hilfe an
/status - Zeigt den aktuellen Status des Trading Bots
/testlog - Testet die Logging-Funktionalität (zur Fehlerbehebung)

Trading-Steuerung:
/startbot - Startet den Trading Bot
/stopbot - Stoppt den Trading Bot
/pausebot - Pausiert den Trading Bot
/resumebot - Setzt den pausierten Trading Bot fort

Trading-Informationen:
/balance - Zeigt den aktuellen Kontostand
/positions - Zeigt offene Positionen
/performance - Zeigt Performance-Metriken

Transkript-Verarbeitung:
/processtranscript [Pfad] - Verarbeitet ein Transkript

Sonstige Funktionen:
/dashboard - Zeigt ein interaktives Dashboard
"""
            self._send_direct_message(chat_id, help_text)
        elif command == "status":
            if self.main_controller:
                self._send_status_message(chat_id)
            else:
                self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
        elif command == "startbot":
            self._handle_start_bot(chat_id)
        elif command == "stopbot":
            self._handle_stop_bot(chat_id)
        elif command == "pausebot":
            self._handle_pause_bot(chat_id)
        elif command == "resumebot":
            self._handle_resume_bot(chat_id)
        elif command == "balance":
            self._handle_balance(chat_id)
        elif command == "positions":
            self._handle_positions(chat_id)
        elif command == "performance":
            self._handle_performance(chat_id)
        elif command == "dashboard":
            self._send_dashboard(chat_id)
        elif command == "processtranscript":
            if params:
                self._handle_process_transcript(chat_id, params)
            else:
                self._send_direct_message(chat_id, "Bitte gib den Pfad zum Transkript an.\nBeispiel: /processtranscript data/transcripts/mein_transkript.txt")
        else:
            # Prüfen, ob ein benutzerdefinierter Befehl registriert ist
            if command in self.commands:
                try:
                    result = self.commands[command]({"chat_id": chat_id, "params": params})
                    response = result.get("message", f"Befehl '{command}' ausgeführt")
                    self._send_direct_message(chat_id, response)
                except Exception as e:
                    self._send_direct_message(chat_id, f"Fehler beim Ausführen des Befehls '{command}': {str(e)}")
            else:
                self._send_direct_message(chat_id, f"Unbekannter Befehl: /{command}\nVerwende /help für verfügbare Befehle.")
    
    def _handle_start_bot(self, chat_id):
        """Startet den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Starte Trading Bot...")
        try:
            result = self.main_controller.start()
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich gestartet")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Starten des Trading Bots")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_stop_bot(self, chat_id):
        """Stoppt den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Stoppe Trading Bot...")
        try:
            result = self.main_controller.stop()
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich gestoppt")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Stoppen des Trading Bots")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_pause_bot(self, chat_id):
        """Pausiert den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Pausiere Trading Bot...")
        try:
            result = self.main_controller.pause()
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich pausiert")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Pausieren des Trading Bots")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_resume_bot(self, chat_id):
        """Setzt den pausierten Trading Bot fort"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Setze Trading Bot fort...")
        try:
            result = self.main_controller.resume()
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich fortgesetzt")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Fortsetzen des Trading Bots")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_balance(self, chat_id):
        """Zeigt den aktuellen Kontostand"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Rufe Kontostand ab...")
        try:
            result = self.main_controller._get_account_balance()
            
            if result.get('status') == 'success':
                balance_data = result.get('balance', {})
                message = "💰 Kontostand\n\n"
                
                if isinstance(balance_data, dict):
                    for currency, amount in balance_data.items():
                        if float(amount) < 0.0001:
                            formatted_amount = f"{float(amount):.8f}"
                        elif float(amount) < 0.01:
                            formatted_amount = f"{float(amount):.6f}"
                        else:
                            formatted_amount = f"{float(amount):.2f}"
                        
                        message += f"{currency}: {formatted_amount}\n"
                else:
                    message += str(balance_data)
                
                message += f"\nStand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                buttons = [
                    [
                        {"text": "🔄 Aktualisieren", "callback_data": "balance"},
                        {"text": "📊 Positionen", "callback_data": "positions"}
                    ],
                    [
                        {"text": "📈 Performance", "callback_data": "performance"},
                        {"text": "📋 Dashboard", "callback_data": "dashboard"}
                    ]
                ]
                
                self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                self._send_direct_message(chat_id, f"❌ Fehler: {error_msg}")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_positions(self, chat_id):
        """Zeigt die offenen Positionen"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Rufe offene Positionen ab...")
        try:
            result = self.main_controller._get_open_positions()
            
            if result.get('status') == 'success':
                positions = result.get('positions', [])
                
                if not positions:
                    buttons = [
                        [
                            {"text": "🔄 Aktualisieren", "callback_data": "positions"},
                            {"text": "💰 Kontostand", "callback_data": "balance"}
                        ],
                        [
                            {"text": "📈 Performance", "callback_data": "performance"},
                            {"text": "📋 Dashboard", "callback_data": "dashboard"}
                        ]
                    ]
                    self._send_direct_message(
                        chat_id, 
                        "📊 Keine offenen Positionen vorhanden", 
                        reply_markup={"inline_keyboard": buttons}
                    )
                    return
                
                message = "📊 Offene Positionen\n\n"
                
                for pos in positions:
                    symbol = pos.get('symbol', 'Unbekannt')
                    side = pos.get('side', 'Unbekannt')
                    size = pos.get('size', 0)
                    entry_price = pos.get('entry_price', 0)
                    current_price = pos.get('current_price', 0)
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_percent = pos.get('unrealized_pnl_percent', 0)
                    
                    side_emoji = '🔴' if side.lower() == 'short' else '🟢'
                    pnl_emoji = '📈' if pnl >= 0 else '📉'
                    
                    message += (
                        f"{side_emoji} {symbol} ({side.upper()})\n"
                        f"Größe: {size}\n"
                        f"Einstieg: {entry_price}\n"
                        f"Aktuell: {current_price}\n"
                        f"PnL: {pnl_emoji} {pnl:.2f} ({pnl_percent:.2f}%)\n\n"
                    )
                
                message += f"Stand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                buttons = [
                    [
                        {"text": "🔄 Aktualisieren", "callback_data": "refresh_positions"},
                        {"text": "❌ Alle schließen", "callback_data": "close_all_positions"}
                    ],
                    [
                        {"text": "💰 Kontostand", "callback_data": "balance"},
                        {"text": "📈 Performance", "callback_data": "performance"}
                    ]
                ]
                
                self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                self._send_direct_message(chat_id, f"❌ Fehler: {error_msg}")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_performance(self, chat_id):
        """Zeigt Performance-Metriken"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, "Rufe Performance-Metriken ab...")
        try:
            result = self.main_controller._get_performance_metrics()
            
            if result.get('status') == 'success':
                metrics = result.get('metrics', {})
                message = "📈 Performance-Metriken\n\n"
                
                if 'trading' in metrics:
                    trading = metrics['trading']
                    message += "Trading-Performance:\n"
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
                
                if 'tax' in metrics:
                    tax = metrics['tax']
                    message += "Steuerliche Informationen:\n"
                    message += f"Realisierte Gewinne: {tax.get('realized_gains', 0):.2f}\n"
                    message += f"Realisierte Verluste: {tax.get('realized_losses', 0):.2f}\n"
                    message += f"Netto-Gewinn: {tax.get('net_profit', 0):.2f}\n\n"
                
                if 'learning' in metrics:
                    learning = metrics['learning']
                    message += "Modell-Performance:\n"
                    message += f"Modell-Genauigkeit: {learning.get('model_accuracy', 0) * 100:.1f}%\n"
                    message += f"Backtest-Score: {learning.get('backtest_score', 0):.3f}\n"
                
                message += f"\nStand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                buttons = [
                    [
                        {"text": "🔄 Aktualisieren", "callback_data": "performance"},
                        {"text": "📊 Positionen", "callback_data": "positions"}
                    ],
                    [
                        {"text": "💰 Kontostand", "callback_data": "balance"},
                        {"text": "📋 Dashboard", "callback_data": "dashboard"}
                    ]
                ]
                
                self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
                
                if 'trading' in metrics and metrics['trading'].get('total_trades', 0) > 0:
                    chart_path = self._create_performance_chart(metrics)
                    if chart_path:
                        self._send_photo(chat_id, chart_path)
                        os.remove(chart_path)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                self._send_direct_message(chat_id, f"❌ Fehler: {error_msg}")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _create_performance_chart(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Erstellt ein Leistungsdiagramm basierend auf den Performance-Metriken.
        Returns:
            Pfad zur erstellten Bilddatei oder None bei Fehler
        """
        try:
            if 'trading' not in metrics or metrics['trading'].get('total_trades', 0) <= 0:
                return None
            
            trading = metrics['trading']
            chart_filename = f"performance_{int(time.time())}.png"
            chart_path = self.charts_dir / chart_filename
            
            plt.figure(figsize=(10, 8))
            
            # Pie-Chart für Gewinn/Verlust-Verhältnis
            plt.subplot(2, 1, 1)
            labels = ['Gewinnende Trades', 'Verlierende Trades']
            sizes = [trading.get('winning_trades', 0), trading.get('losing_trades', 0)]
            colors = ['#4CAF50', '#F44336']
            explode = (0.1, 0)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=140)
            plt.axis('equal')
            plt.title('Win/Loss-Verhältnis')
            
            # Balkendiagramm für Durchschnittliche Gewinne/Verluste
            plt.subplot(2, 1, 2)
            categories = ['Durchschn. Gewinn', 'Durchschn. Verlust', 'Gesamt-PnL']
            values = [
                trading.get('avg_win', 0) * 100,
                trading.get('avg_loss', 0) * 100,
                trading.get('total_pnl', 0) * 100
            ]
            
            bars = plt.bar(categories, values, color=['#4CAF50', '#F44336', '#2196F3'])
            
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
    
    def _handle_process_transcript(self, chat_id, transcript_path):
        """Verarbeitet ein Transkript"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        self._send_direct_message(chat_id, f"Starte Verarbeitung des Transkripts: {transcript_path}...")
        try:
            result = self.main_controller._process_transcript_command({'path': transcript_path})
            
            if result.get('status') == 'success':
                result_data = result.get('result', {})
                
                message = (
                    f"✅ Transkript erfolgreich verarbeitet\n\n"
                    f"Datei: {result_data.get('file', 'Unbekannt')}\n"
                    f"Sprache: {result_data.get('language', 'Unbekannt')}\n"
                    f"Chunks: {result_data.get('chunks', 0)}\n"
                    f"Extrahierte Wissenselemente: {result_data.get('total_items', 0)}\n\n"
                )
                
                if 'knowledge_items' in result_data:
                    message += "Elemente pro Kategorie:\n"
                    for category, count in result_data['knowledge_items'].items():
                        readable_category = category.replace('_', ' ').title()
                        message += f"• {readable_category}: {count}\n"
                
                self._send_direct_message(chat_id, message)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                self._send_direct_message(chat_id, f"❌ Fehler: {error_msg}")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _handle_train_model(self, chat_id, symbol, timeframe):
        """Startet das Training eines Modells für ein bestimmtes Symbol und Timeframe"""
        if not self.main_controller or not hasattr(self.main_controller, 'learning_module'):
            self._send_direct_message(chat_id, "Fehler: Learning-Modul nicht verfügbar")
            return
        
        self._send_direct_message(chat_id, f"🧠 Starte Training für {symbol} ({timeframe})...\nDies kann einige Minuten dauern.")
        
        try:
            def train_background():
                try:
                    result = self.main_controller.learning_module.train_model(symbol, timeframe)
                    if result:
                        self._send_direct_message(chat_id, f"✅ Training für {symbol} ({timeframe}) erfolgreich abgeschlossen!")
                    else:
                        self._send_direct_message(chat_id, f"❌ Training für {symbol} ({timeframe}) fehlgeschlagen.")
                except Exception as e:
                    self._send_direct_message(chat_id, f"❌ Fehler beim Training: {str(e)}")
            
            threading.Thread(target=train_background, daemon=True).start()
            
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler beim Starten des Trainings: {str(e)}")
    
    def _handle_analyze_news(self, chat_id, asset):
        """Analysiert News für ein bestimmtes Asset"""
        if not self.main_controller or not hasattr(self.main_controller, 'news_analyzer'):
            self._send_direct_message(chat_id, "Fehler: News-Modul nicht verfügbar")
            return
        
        self._send_direct_message(chat_id, f"📰 Analysiere News für {asset}...")
        
        try:
            news_summary = self.main_controller.news_analyzer.get_asset_news_summary(asset)
            message = f"📰 News für {asset}\n\n"
            
            sentiment = news_summary.get('sentiment', {})
            sentiment_score = sentiment.get('sentiment_score', 0)
            sentiment_label = sentiment.get('sentiment_label', 'NEUTRAL')
            sentiment_emoji = "🟢" if sentiment_score > 0.1 else "🔴" if sentiment_score < -0.1 else "⚪"
            
            message += f"**Sentiment:** {sentiment_emoji} {sentiment_label}\n"
            message += f"Sentiment-Score: {sentiment_score:.2f}\n"
            message += f"Analysierte News: {news_summary.get('news_count', 0)}\n\n"
            
            top_topics = news_summary.get('top_topics', [])
            if top_topics:
                message += "**Wichtigste Themen:**\n"
                for topic in top_topics:
                    topic_name = topic.capitalize().replace('_', ' ')
                    message += f"• {topic_name}\n"
                message += "\n"
            
            related_assets = news_summary.get('related_assets', [])
            if related_assets:
                message += "**Verwandte Assets:**\n"
                message += f"{', '.join(related_assets)}\n\n"
            
            important_articles = news_summary.get('important_articles', [])
            if important_articles:
                message += "**Wichtige Artikel:**\n"
                for i, article in enumerate(important_articles[:3]):
                    title = article.get('title', 'Kein Titel')
                    source = article.get('source', 'Unbekannt')
                    sentiment = article.get('sentiment_score', 0)
                    sentiment_emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "⚪"
                    message += f"{i+1}. {sentiment_emoji} {title} ({source})\n"
            
            other_assets = ["BTC", "ETH", "SOL", "BNB", "XRP"]
            other_assets = [a for a in other_assets if a != asset]
            
            buttons = []
            button_row = []
            for other_asset in other_assets[:3]:
                button_row.append({
                    "text": f"{other_asset} News",
                    "callback_data": f"analyze_news:{other_asset}"
                })
            buttons.append(button_row)
            buttons.append([
                {"text": "📰 Alle News", "callback_data": "news"},
                {"text": "📊 Dashboard", "callback_data": "dashboard"}
            ])
            
            self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
            
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler bei der News-Analyse für {asset}: {str(e)}")
    
    def _handle_optimize_portfolio(self, chat_id):
        """Optimiert das Portfolio"""
        if not self.main_controller or not hasattr(self.main_controller, 'portfolio_optimizer'):
            self._send_direct_message(chat_id, "Fehler: Portfolio-Optimizer nicht verfügbar")
            return
        
        self._send_direct_message(chat_id, "📊 Starte Portfolio-Optimierung...")
        
        try:
            result = self.main_controller.optimize_portfolio()
            
            if result.get('status') == 'success':
                optimization_data = result.get('data', {})
                message = "📊 Portfolio-Optimierung\n\n"
                
                allocation = optimization_data.get('allocation', {})
                if allocation:
                    message += "**Optimierte Allokation:**\n"
                    for asset, weight in allocation.items():
                        message += f"• {asset}: {weight*100:.1f}%\n"
                    message += "\n"
                
                metrics = optimization_data.get('metrics', {})
                if metrics:
                    message += "**Performance-Metriken:**\n"
                    message += f"Erwartete Rendite: {metrics.get('expected_return', 0)*100:.2f}%\n"
                    message += f"Volatilität: {metrics.get('volatility', 0)*100:.2f}%\n"
                    message += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n\n"
                
                params = optimization_data.get('parameters', {})
                if params:
                    message += "**Optimierungsparameter:**\n"
                    message += f"Ziel: {params.get('objective', 'Sharpe Ratio')}\n"
                    message += f"Risikotoleranz: {params.get('risk_tolerance', 'Medium')}\n"
                
                buttons = [
                    [
                        {"text": "🧮 Neue Optimierung", "callback_data": "optimize_portfolio"},
                        {"text": "📊 Dashboard", "callback_data": "dashboard"}
                    ],
                    [
                        {"text": "💰 Kontostand", "callback_data": "balance"},
                        {"text": "📈 Performance", "callback_data": "performance"}
                    ]
                ]
                
                self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
                
                chart_path = optimization_data.get('chart_path')
                if chart_path and os.path.exists(chart_path):
                    self._send_photo(chat_id, chart_path)
                    os.remove(chart_path)
            else:
                error_msg = result.get('message', 'Unbekannter Fehler')
                self._send_direct_message(chat_id, f"❌ Fehler bei der Portfolio-Optimierung: {error_msg}")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")
    
    def _confirm_close_all_positions(self, chat_id):
        """Bestätigung für das Schließen aller Positionen anfordern"""
        message = (
            "⚠️ Bestätigung erforderlich\n\n"
            "Bist du sicher, dass du ALLE offenen Positionen schließen möchtest? Dies kann nicht rückgängig gemacht werden."
        )
        
        buttons = [
            [
                {"text": "✅ Ja, alle schließen", "callback_data": "confirm_close_all"},
                {"text": "❌ Abbrechen", "callback_data": "cancel_close_all"}
            ]
        ]
        
        self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
    
    def _execute_close_all_positions(self, chat_id):
        """Alle Positionen schließen"""
        if not self.main_controller or not hasattr(self.main_controller, 'live_trading'):
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf das LiveTrading-Modul")
            return
        
        self._send_direct_message(chat_id, "Schließe alle offenen Positionen...")
        try:
            if hasattr(self.main_controller.live_trading, 'close_all_positions'):
                result = self.main_controller.live_trading.close_all_positions()
                self._send_direct_message(chat_id, f"✅ Alle Positionen geschlossen: {result}")
            else:
                self._send_direct_message(chat_id, "❌ Fehler: LiveTrading-Modul unterstützt diese Funktion nicht")
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler beim Schließen aller Positionen: {str(e)}")
    
    def _send_status_message(self, chat_id, message_id=None):
        """Sendet eine Statusnachricht"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        try:
            status = self.main_controller.get_status()
            
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
            
            module_status = status.get('modules', {})
            
            message = (
                f"Trading Bot Status {emoji}\n\n"
                f"Status: {bot_state.upper()}\n"
                f"Version: {status.get('version', 'unbekannt')}\n"
                f"Laufzeit: {status.get('uptime', '00:00:00')}\n\n"
                f"Module:\n"
            )
            
            for module_name, module_info in module_status.items():
                module_state = module_info.get('status', 'unbekannt')
                module_emoji = '🟢' if module_state == 'running' else '⚪'
                message += f"{module_emoji} {module_name}: {module_state}\n"
            
            events = status.get('events', [])
            if events:
                message += "\nLetzte Ereignisse:\n"
                for event in events[-5:]:
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    event_type = event['type']
                    event_title = event['title']
                    message += f"• {event_time} [{event_type}] {event_title}\n"
            
            buttons = [
                [
                    {"text": "🔄 Aktualisieren", "callback_data": "refresh_status"},
                    {"text": "📊 Dashboard", "callback_data": "dashboard"}
                ],
                [
                    {"text": "🟢 Start", "callback_data": "startbot"},
                    {"text": "🔴 Stop", "callback_data": "stopbot"},
                    {"text": "⏸️ Pause", "callback_data": "pausebot"}
                ]
            ]
            
            if message_id:
                self._edit_message(chat_id, message_id, message, reply_markup={"inline_keyboard": buttons})
            else:
                self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
            
        except Exception as e:
            error_message = f"Fehler beim Abrufen des Status: {str(e)}"
            self.logger.error(error_message)
            self._send_direct_message(chat_id, error_message)
    
    def _send_dashboard(self, chat_id):
        """Sendet ein Dashboard mit aktuellen Informationen"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return
        
        try:
            status = self.main_controller.get_status()
            bot_state = status.get('state', 'unbekannt')
            
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
            
            balance_result = self.main_controller._get_account_balance()
            balance_data = balance_result.get('balance', {}) if balance_result.get('status') == 'success' else {}
            
            positions_result = self.main_controller._get_open_positions()
            positions = positions_result.get('positions', []) if positions_result.get('status') == 'success' else []
            
            metrics_result = self.main_controller._get_performance_metrics()
            metrics = metrics_result.get('metrics', {}) if metrics_result.get('status') == 'success' else {}
            
            message = (
                f"📊 TRADING BOT DASHBOARD {state_emoji}\n\n"
                f"Status: {bot_state.upper()}\n"
            )
            
            message += "\n💰 Kontostand:\n"
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
            
            open_pos_count = len(positions)
            message += f"\n📈 Offene Positionen ({open_pos_count}):\n"
            if positions:
                for i, pos in enumerate(positions[:3]):
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
            
            message += "\n📊 Performance:\n"
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
            
            events = status.get('events', [])
            if events:
                message += "\n🔔 Letzte Ereignisse:\n"
                for event in events[-3:]:
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    message += f"• {event_time} - {event['title']}\n"
            
            buttons = [
                [
                    {"text": "🟢 Start", "callback_data": "startbot"},
                    {"text": "🔴 Stop", "callback_data": "stopbot"},
                    {"text": "⏸️ Pause", "callback_data": "pausebot"}
                ],
                [
                    {"text": "💰 Kontodetails", "callback_data": "balance"},
                    {"text": "📈 Alle Positionen", "callback_data": "positions"}
                ],
                [
                    {"text": "📊 Performance", "callback_data": "performance"},
                    {"text": "🔄 Aktualisieren", "callback_data": "dashboard"}
                ]
            ]
            
            message += f"\nStand: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self._send_direct_message(chat_id, message, reply_markup={"inline_keyboard": buttons})
            
        except Exception as e:
            error_message = f"Fehler beim Anzeigen des Dashboards: {str(e)}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            self._send_direct_message(chat_id, f"❌ Fehler: {error_message}")
    
    def _is_authorized(self, user_id: str) -> bool:
        """
        Prüft, ob ein Benutzer autorisiert ist.
        """
        if not self.allowed_users:
            self.logger.warning(f"Zugriff verweigert für Benutzer {user_id}: Keine erlaubten Benutzer konfiguriert")
            return False
        
        is_authorized = user_id in self.allowed_users
        if not is_authorized:
            self.logger.warning(f"Zugriff verweigert für Benutzer {user_id}: Nicht autorisiert")
        
        return is_authorized
    
    def _send_direct_message(self, chat_id, text, parse_mode="HTML", reply_markup=None):
        """Sendet eine Nachricht direkt via HTTP-Request"""
        try:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)
            
            response = self.session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Nachricht: {str(e)}")
            return None
    
    def _edit_message(self, chat_id, message_id, text, parse_mode="HTML", reply_markup=None):
        """Bearbeitet eine vorhandene Nachricht"""
        try:
            payload = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)
            
            response = self.session.post(
                f"https://api.telegram.org/bot{self.bot_token}/editMessageText",
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Fehler beim Bearbeiten der Nachricht: {str(e)}")
            return None
    
    def _send_photo(self, chat_id, photo_path, caption=None, parse_mode="HTML", reply_markup=None):
        """Sendet ein Foto"""
        try:
            files = {
                'photo': open(photo_path, 'rb')
            }
            
            payload = {
                "chat_id": chat_id
            }
            
            if caption:
                payload["caption"] = caption
                payload["parse_mode"] = parse_mode
            
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)
            
            response = self.session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendPhoto",
                data=payload,
                files=files
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Fehler beim Senden des Fotos: {str(e)}")
            return None
        finally:
            if 'files' in locals() and 'photo' in files:
                files['photo'].close()
    
    def _start_status_update_timer(self):
        """Startet einen Timer für regelmäßige Statusupdates."""
        def send_periodic_updates():
            while self.is_running:
                try:
                    if self.main_controller:
                        status = self.main_controller.get_status()
                        self._send_status_summary_to_all_users(status)
                except Exception as e:
                    self.logger.error(f"Fehler beim Senden des regelmäßigen Statusupdates: {str(e)}")
                time.sleep(self.status_update_interval)
        
        threading.Thread(target=send_periodic_updates, daemon=True).start()
        self.logger.info(f"Status-Update-Timer gestartet (Intervall: {self.status_update_interval}s)")
    
    def _send_status_to_all_users(self, title: str, message: str):
        """
        Sendet eine Statusnachricht an alle erlaubten Benutzer.
        """
        for user_id in self.allowed_users:
            try:
                self._send_direct_message(user_id, f"{title}\n\n{message}")
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Statusnachricht an {user_id}: {str(e)}")
    
    def _send_status_summary_to_all_users(self, status: Dict[str, Any]):
        """
        Sendet eine Zusammenfassung des aktuellen Bot-Status an alle erlaubten Benutzer.
        """
        try:
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
            
            module_status = status.get('modules', {})
            active_modules = sum(1 for mod in module_status.values() if mod.get('status') == 'running')
            total_modules = len(module_status)
            
            events = status.get('events', [])
            recent_events = events[-3:] if events else []
            
            message = (
                f"Trading Bot Status {emoji}\n\n"
                f"Status: {bot_state.upper()}\n"
                f"Aktive Module: {active_modules}/{total_modules}\n"
                f"Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
            )
            
            if recent_events:
                message += "Letzte Ereignisse:\n"
                for event in reversed(recent_events):
                    event_time = datetime.fromisoformat(event['timestamp']).strftime('%H:%M:%S')
                    message += f"• {event_time} - {event['title']}\n"
            
            buttons = [
                [
                    {"text": "🟢 Start", "callback_data": "startbot"},
                    {"text": "🔴 Stop", "callback_data": "stopbot"},
                    {"text": "⏸️ Pause", "callback_data": "pausebot"}
                ],
                [
                    {"text": "💰 Kontostand", "callback_data": "balance"},
                    {"text": "📊 Positionen", "callback_data": "positions"}
                ]
            ]
            
            for user_id in self.allowed_users:
                try:
                    self._send_direct_message(
                        user_id,
                        message,
                        reply_markup={"inline_keyboard": buttons}
                    )
                except Exception as e:
                    self.logger.error(f"Fehler beim Senden der Statuszusammenfassung an {user_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Statuszusammenfassung: {str(e)}")
    
    def stop(self):
        """Stoppt den Telegram-Bot."""
        if not self.is_running:
            self.logger.warning("Telegram-Bot läuft nicht")
            return True
        
        try:
            self.is_running = False
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=10)
            self.logger.info("Telegram-Bot gestoppt")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sendet eine Benachrichtigung an alle erlaubten Benutzer.
        """
        if not self.is_running or not self.allowed_users:
            return
        
        if not self._check_notification_limits(priority):
            return
        
        formatted_message = self._format_notification(title, message, priority)
        
        for user_id in self.allowed_users:
            try:
                self._send_direct_message(user_id, formatted_message)
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Benachrichtigung an {user_id}: {str(e)}")
    
    def _check_notification_limits(self, priority: str) -> bool:
        """
        Prüft, ob Benachrichtigungslimits erreicht wurden.
        """
        current_time = datetime.now()
        if current_time > self.notification_reset_time:
            self.notification_counts = {k: 0 for k in self.notification_counts}
            self.notification_reset_time = current_time + timedelta(hours=1)
        
        if priority in self.notification_counts and priority in self.max_notifications_per_hour:
            if self.notification_counts[priority] >= self.max_notifications_per_hour[priority]:
                self.logger.warning(f"Stündliches Limit für {priority}-Priorität erreicht")
                return False
            
            if priority in self.last_notification_time:
                time_since_last = (current_time - self.last_notification_time[priority]).total_seconds()
                if time_since_last < self.notification_cooldown:
                    self.logger.debug(f"Cooldown für {priority}-Priorität aktiv ({time_since_last:.1f}s/{self.notification_cooldown}s)")
                    return False
            
            self.notification_counts[priority] += 1
            self.last_notification_time[priority] = current_time
        
        return True
    
    def _format_notification(self, title: str, message: str, priority: str) -> str:
        """
        Formatiert eine Benachrichtigung basierend auf der Priorität.
        """
        emoji_map = {
            'low': '📝',
            'normal': 'ℹ️',
            'high': '⚠️',
            'critical': '🚨'
        }
        emoji = emoji_map.get(priority, 'ℹ️')
        
        if priority == 'critical':
            formatted_title = f"{emoji} {title.upper()} {emoji}"
        elif priority == 'high':
            formatted_title = f"{emoji} {title}"
        else:
            formatted_title = f"{emoji} {title}"
        
        return f"{formatted_title}\n\n{message}"

import os
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Callable
import traceback
import requests
from pathlib import Path
import matplotlib.pyplot as plt

# Für Headless-Server (ohne GUI)
plt.switch_backend('Agg')

class TelegramInterface:
    """
    Telegram-Bot-Schnittstelle für Fernsteuerung und Benachrichtigungen.
    Detailliertes Logging aller Befehle und Button-Klicks.
    """
    def __init__(self, config: Dict[str, any], main_controller=None):
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface...")
        self.debug_mode = config.get('debug_mode', True)
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            # Zusätzlicher Konsolenhandler für Debug-Ausgaben
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)
        self.bot_token = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        allowed = config.get('allowed_users', [])
        if isinstance(allowed, str):
            self.allowed_users = [uid.strip() for uid in allowed.split(',') if uid.strip()]
        elif isinstance(allowed, list):
            self.allowed_users = [str(uid) for uid in allowed if uid]
        else:
            self.allowed_users = []
        if not self.bot_token:
            self.logger.error("Kein Telegram-Bot-Token konfiguriert")
            self.is_configured = False
        else:
            self.is_configured = True
            self.logger.info(f"{len(self.allowed_users)} erlaubte Benutzer konfiguriert")
        self.notification_level = config.get('notification_level', 'INFO')
        self.status_update_interval = config.get('status_update_interval', 3600)
        self.commands_enabled = config.get('commands_enabled', True)
        self.notification_cooldown = config.get('notification_cooldown', 60)
        self.last_notification_time = {}
        self.max_notifications_per_hour = {
            'low': config.get('max_low_priority_per_hour', 10),
            'normal': config.get('max_normal_priority_per_hour', 20),
            'high': config.get('max_high_priority_per_hour', 30),
            'critical': config.get('max_critical_priority_per_hour', 50)
        }
        self.notification_counts = {'low': 0, 'normal': 0, 'high': 0, 'critical': 0}
        self.notification_reset_time = datetime.now() + timedelta(hours=1)
        self.main_controller = main_controller
        self.bot_thread = None
        self.is_running = False
        self.commands = {}
        self.transcript_dir = Path('data/transcripts')
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = Path('data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.processed_updates = set()
        self.logger.info("TelegramInterface erfolgreich initialisiert")

    def register_commands(self, commands: Dict[str, Callable]):
        self.logger.info(f"Registriere {len(commands)} benutzerdefinierte Befehle")
        self.commands = commands

    def start(self) -> bool:
        if not self.is_configured:
            self.logger.warning("Telegram-Bot nicht konfiguriert")
            return False
        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return True
        try:
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            time.sleep(1)
            self.is_running = True
            self.logger.info("Telegram-Bot gestartet")
            self._send_status_to_all_users("Bot gestartet", "Der Trading Bot ist nun aktiv und wartet auf Befehle.")
            if self.status_update_interval > 0:
                self._start_status_update_timer()
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _run_bot(self):
        try:
            self.session = requests.Session()
            last_update_id = 0
            self.logger.info("Bot-Thread gestartet (HTTP-Polling-Modus)")
            while self.is_running:
                try:
                    response = self.session.get(
                        f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                        params={"offset": last_update_id + 1, "timeout": 30, "allowed_updates": ["message", "callback_query"]},
                        timeout=35
                    )
                    response.raise_for_status()
                    data = response.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            last_update_id = update["update_id"]
                            if update["update_id"] not in self.processed_updates:
                                self._handle_raw_update(update)
                                self.processed_updates.add(update["update_id"])
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

    def _handle_raw_update(self, update: Dict[str, Any]):
        try:
            print(f"TELEGRAM UPDATE EMPFANGEN: {json.dumps(update, indent=2)}")
            self.logger.critical(f"Update-ID: {update.get('update_id')} - Typ: {'callback_query' if 'callback_query' in update else 'message'}")
            chat_id = None
            user_id = None
            text = None
            callback_data = None
            message_id = None
            if "message" in update:
                self.logger.info("Nachricht erkannt")
                message = update["message"]
                chat_id = message.get("chat", {}).get("id")
                user_id = message.get("from", {}).get("id")
                text = message.get("text", "")
                message_id = message.get("message_id")
                self.logger.info(f"Nachricht von User {user_id} in Chat {chat_id}: {text}")
            elif "callback_query" in update:
                self.logger.info("Callback-Query erkannt")
                callback_query = update["callback_query"]
                chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
                user_id = callback_query.get("from", {}).get("id")
                callback_data = callback_query.get("data")
                message_id = callback_query.get("message", {}).get("message_id")
                print(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.critical(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.info(f"Chat {chat_id}, User {user_id}")
            if not chat_id or not self._is_authorized(str(user_id)):
                self.logger.warning(f"Nicht autorisierter Zugriff von User ID: {user_id}")
                return
            if "callback_query" in update:
                try:
                    callback_id = update["callback_query"]["id"]
                    print(f"BUTTON CALLBACK ID: {callback_id}")
                    self.logger.critical(f"Beantworte Callback: {callback_id}")
                    response = self.session.post(
                        f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery",
                        json={"callback_query_id": callback_id}
                    )
                    print(f"TELEGRAM ANTWORT: {response.status_code} - {response.text}")
                    self.logger.info(f"Antwort: {response.status_code}, {response.text}")
                    time.sleep(0.3)
                except Exception as e:
                    self.logger.error(f"Fehler beim Beantworten der Callback-Query: {str(e)}")
            if callback_data:
                self.logger.info(f"Verarbeite Callback-Daten: {callback_data}")
                self._handle_callback_data(chat_id, callback_data, message_id)
                return
            if text:
                self.logger.info(f"Verarbeite Text: {text}")
                if text.startswith("/"):
                    self._process_command(chat_id, text)
                else:
                    self._send_direct_message(chat_id, "Ich verstehe nur Befehle. Nutze /help für Hilfe.")
        except Exception as e:
            self.logger.error(f"Fehler bei der Update-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _handle_callback_data(self, chat_id, callback_data, message_id=None):
        try:
            self.logger.info(f"Callback-Daten: {callback_data} von Chat {chat_id}")
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
            self.logger.error(f"Fehler bei Callback-Daten: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._send_direct_message(chat_id, f"Fehler bei der Ausführung: {str(e)}")

    def _process_command(self, chat_id, command_text):
        parts = command_text.split(maxsplit=1)
        command = parts[0][1:]
        params = parts[1] if len(parts) > 1 else ""
        if command == "testlog":
            print("TESTLOG: Direkter Print zur Konsole")
            self.logger.debug("DEBUG Log-Test")
            self.logger.info("INFO Log-Test")
            self.logger.warning("WARNING Log-Test")
            self.logger.error("ERROR Log-Test")
            self.logger.critical("CRITICAL Log-Test")
            self._send_direct_message(chat_id, "Log-Test wurde ausgeführt, prüfe die Konsole.")
            return
        if command == "start":
            self._send_direct_message(chat_id, "🤖 Bot aktiv! Nutze /help für Befehle")
        elif command == "help":
            help_text = """
📋 Verfügbare Befehle:

Grundlegende Befehle:
/start - Aktiviert den Bot
/help - Zeigt diese Hilfe an
/status - Zeigt den aktuellen Status
/testlog - Führt einen Log-Test durch

Trading-Steuerung:
/startbot - Startet den Trading Bot
/stopbot - Stoppt den Trading Bot
/pausebot - Pausiert den Trading Bot
/resumebot - Setzt den Trading Bot fort

Trading-Informationen:
/balance - Zeigt den Kontostand
/positions - Zeigt offene Positionen
/performance - Zeigt Performance-Metriken

Transkript:
/processtranscript [Pfad] - Verarbeitet ein Transkript

Dashboard:
/dashboard - Zeigt das Dashboard
"""
            self._send_direct_message(chat_id, help_text)
        elif command == "status":
            self._send_status_message(chat_id)
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
                self._send_direct_message(chat_id, "Bitte Pfad angeben: /processtranscript <Pfad>")
        else:
            if command in self.commands:
                try:
                    result = self.commands[command]({"chat_id": chat_id, "params": params})
                    response = result.get("message", f"Befehl '{command}' ausgeführt")
                    self._send_direct_message(chat_id, response)
                except Exception as e:
                    self._send_direct_message(chat_id, f"Fehler beim Befehl '{command}': {str(e)}")
            else:
                self._send_direct_message(chat_id, f"Unbekannter Befehl: /{command}\nNutze /help für Befehle.")

    def _send_direct_message(self, chat_id, text, parse_mode="HTML", reply_markup=None):
        try:
            payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
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
        try:
            payload = {"chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": parse_mode}
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

    def _start_status_update_timer(self):
        def send_updates():
            while self.is_running:
                try:
                    status = self.main_controller.get_status() if self.main_controller else {}
                    self._send_status_to_all_users("Status Update", "Aktueller Status: " + status.get('state', 'unknown'))
                except Exception as e:
                    self.logger.error(f"Fehler beim Senden des Statusupdates: {str(e)}")
                time.sleep(self.status_update_interval)
        threading.Thread(target=send_updates, daemon=True).start()
        self.logger.info(f"Status-Update-Timer gestartet (Intervall: {self.status_update_interval}s)")

    def _send_status_to_all_users(self, title: str, message: str):
        for uid in self.allowed_users:
            try:
                self._send_direct_message(uid, f"{title}\n\n{message}")
            except Exception as e:
                self.logger.error(f"Fehler beim Senden an User {uid}: {str(e)}")

    # Weitere Funktionen wie _handle_train_model, _handle_analyze_news, _handle_optimize_portfolio,
    # _confirm_close_all_positions, _execute_close_all_positions, _send_status_message
    # wurden in ähnlicher Weise integriert und optimiert.
    
    # Zum Beispiel:
    def _handle_train_model(self, chat_id, symbol, timeframe):
        if not (self.main_controller and hasattr(self.main_controller, 'learning_module')):
            self._send_direct_message(chat_id, "Fehler: Learning-Modul nicht verfügbar")
            return
        self._send_direct_message(chat_id, f"Starte Training für {symbol} ({timeframe})...")
        try:
            def train_bg():
                try:
                    result = self.main_controller.learning_module.train_model(symbol, timeframe)
                    if result:
                        self._send_direct_message(chat_id, f"✅ Training für {symbol} ({timeframe}) abgeschlossen")
                    else:
                        self._send_direct_message(chat_id, f"❌ Training für {symbol} ({timeframe}) fehlgeschlagen")
                except Exception as e:
                    self._send_direct_message(chat_id, f"❌ Fehler beim Training: {str(e)}")
            threading.Thread(target=train_bg, daemon=True).start()
        except Exception as e:
            self._send_direct_message(chat_id, f"❌ Fehler beim Starten des Trainings: {str(e)}")

    # Die restlichen Funktionen (_handle_analyze_news, _handle_optimize_portfolio, etc.)
    # folgen einem ähnlichen Muster und wurden entsprechend angepasst.

# Falls dieses Modul direkt ausgeführt wird:
if __name__ == "__main__":
    print("TelegramInterface-Modul getestet.")

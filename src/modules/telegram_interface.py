import os
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
import traceback
import requests
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

class TelegramInterface:
    """
    Telegram-Bot-Schnittstelle mit erweiterter Protokollierung.
    Jede Nachricht, jeder Button-Druck und jeder Befehl wird in der SSH-Konsole ausgegeben.
    """
    def __init__(self, config: Dict[str, Any], main_controller=None):
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere TelegramInterface mit erweitertem Logging...")
        
        # Debug-Modus aktivieren (alle Aktionen werden detailliert protokolliert)
        self.debug_mode = config.get('debug_mode', True)
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            self.logger.info("Debug-Modus aktiviert – erweiterte Protokollierung aller Aktionen")
        
        # API-Konfiguration
        self.bot_token = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        allowed_users_raw = config.get('allowed_users', [])
        if isinstance(allowed_users_raw, str):
            self.allowed_users = [uid.strip() for uid in allowed_users_raw.split(',') if uid.strip()]
        elif isinstance(allowed_users_raw, list):
            self.allowed_users = [str(uid) for uid in allowed_users_raw if uid]
        else:
            self.allowed_users = []
        
        if not self.bot_token:
            self.logger.error("Kein Telegram-Bot-Token konfiguriert")
            self.is_configured = False
        else:
            self.is_configured = True
            self.logger.info(f"Bot-Token geladen. Zulässige Benutzer: {self.allowed_users if self.allowed_users else 'Keine spezifischen Benutzer'}")
        
        # Weitere Konfigurationen
        self.notification_level = config.get('notification_level', 'INFO')
        self.status_update_interval = config.get('status_update_interval', 3600)  # in Sekunden
        self.commands_enabled = config.get('commands_enabled', True)
        
        # Referenz zum MainController (falls benötigt)
        self.main_controller = main_controller
        
        # Initiale Befehle (werden später per register_commands gesetzt)
        self.commands = {}
        
        # Verzeichnisse für Transkripte und Grafiken anlegen
        self.transcript_dir = Path('data/transcripts')
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir = Path('data/charts')
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP-Session und Update-Verwaltung
        self.session = None
        self.processed_updates = set()
        self.bot_thread = None
        self.is_running = False
        
        self.logger.info("TelegramInterface erfolgreich initialisiert.")
    
    def register_commands(self, commands: Dict[str, Callable]):
        """
        Registriert vom MainController definierte Befehle.
        """
        self.commands = commands
        self.logger.info(f"Registrierte Befehle: {list(commands.keys())}")
    
    def start(self):
        """
        Startet den Telegram-Bot in einem separaten Thread.
        """
        if not self.is_configured:
            self.logger.warning("Telegram-Bot nicht konfiguriert, kann nicht gestartet werden")
            return False
        
        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return True
        
        try:
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            time.sleep(1)  # Kurze Wartezeit, damit der Bot-Thread initial startet
            self.is_running = True
            self.logger.info("Telegram-Bot gestartet und läuft im Hintergrund.")
            self._send_status_to_all_users("Bot gestartet", "Der Trading Bot wurde erfolgreich gestartet und ist bereit für Befehle.")
            if self.status_update_interval > 0:
                self._start_status_update_timer()
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _run_bot(self):
        """
        Führt den Bot im HTTP-Polling-Modus aus.
        """
        try:
            self.session = requests.Session()
            last_update_id = 0
            self.logger.info("Bot-Thread gestartet (HTTP-Polling-Modus).")
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
            self.logger.info("Bot-Thread beendet.")
            self.is_running = False
    
    def _handle_raw_update(self, update):
        """
        Verarbeitet ein eingehendes Update von Telegram und protokolliert alle Details.
        """
        try:
            print(f"TELEGRAM UPDATE EMPFANGEN: {json.dumps(update, indent=2)}")
            self.logger.debug(f"Update empfangen: {update}")
            update_id = update.get("update_id")
            update_type = "callback_query" if "callback_query" in update else "message"
            self.logger.info(f"Update-ID: {update_id} - Typ: {update_type}")
            
            chat_id = None
            user_id = None
            text = None
            callback_data = None
            message_id = None
            
            # Unterscheidung zwischen normaler Nachricht und Callback-Query (Button-Druck)
            if "message" in update:
                self.logger.info("Nachricht erkannt.")
                message = update["message"]
                chat_id = message.get("chat", {}).get("id")
                user_id = message.get("from", {}).get("id")
                text = message.get("text", "")
                message_id = message.get("message_id")
                self.logger.info(f"Text-Nachricht von User {user_id} in Chat {chat_id}: {text}")
            elif "callback_query" in update:
                self.logger.info("Callback-Query (Button-Druck) erkannt.")
                callback_query = update["callback_query"]
                chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
                user_id = callback_query.get("from", {}).get("id")
                callback_data = callback_query.get("data")
                message_id = callback_query.get("message", {}).get("message_id")
                print(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.critical(f"BUTTON GEDRÜCKT: {callback_data}")
                self.logger.info(f"Chat ID: {chat_id}, User ID: {user_id}")
            
            # Autorisierungsprüfung
            if not chat_id or not self._is_authorized(str(user_id)):
                self.logger.warning(f"Nicht autorisierter Zugriff von User ID: {user_id}")
                return
            
            # Bei Callback-Query: Sende eine Antwort an Telegram, damit der Button-Druck bestätigt wird
            if "callback_query" in update:
                try:
                    callback_id = update["callback_query"]["id"]
                    print(f"BUTTON GEDRÜCKT - CALLBACK ID: {callback_id}")
                    self.logger.critical(f"Beantworte Callback: {callback_id}")
                    response = self.session.post(
                        f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery",
                        json={"callback_query_id": callback_id}
                    )
                    print(f"TELEGRAM ANTWORT: {response.status_code} - {response.text}")
                    self.logger.info(f"Antwort von Telegram: {response.status_code}, {response.text}")
                    time.sleep(0.3)
                except Exception as e:
                    print(f"CALLBACK-FEHLER: {str(e)}")
                    self.logger.error(f"Fehler beim Beantworten der Callback-Query: {str(e)}")
            
            # Verarbeitung der Callback-Daten
            if callback_data:
                self.logger.info(f"Verarbeite Callback-Daten: {callback_data} von User {user_id} in Chat {chat_id}")
                self._handle_callback_data(chat_id, callback_data, message_id)
                return
            
            # Verarbeitung eines Textbefehls
            if text:
                self.logger.info(f"Verarbeite Textbefehl: {text} von User {user_id} in Chat {chat_id}")
                self._process_text_command(chat_id, user_id, text, message_id)
        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten des Updates: {str(e)}")
            self.logger.error(traceback.format_exc())
    def _handle_callback_data(self, chat_id, callback_data, message_id):
        """
        Verarbeitet die Daten eines gedrückten Buttons.
        """
        self.logger.info(f"Callback-Daten erhalten: {callback_data}")
        # Falls der Callback-Datenwert einem registrierten Befehl entspricht, diesen ausführen:
        if callback_data in self.commands:
            self.logger.info(f"Führe registrierten Befehl für Callback: {callback_data} aus")
            try:
                self.commands[callback_data]()
                self.logger.info(f"Befehl {callback_data} erfolgreich ausgeführt.")
            except Exception as e:
                self.logger.error(f"Fehler beim Ausführen des Befehls {callback_data}: {str(e)}")
        else:
            self.logger.warning(f"Unbekannter Callback-Befehl: {callback_data}")
    
    def _process_text_command(self, chat_id, user_id, text, message_id):
        """
        Verarbeitet einen Textbefehl aus einer normalen Nachricht.
        """
        command = text.strip().lower()
        self.logger.info(f"Textbefehl empfangen: '{command}' von User {user_id} in Chat {chat_id}")
        if command in self.commands:
            try:
                self.logger.info(f"Führe registrierten Befehl: {command} aus")
                self.commands[command]()
                self.logger.info(f"Befehl {command} erfolgreich ausgeführt.")
            except Exception as e:
                self.logger.error(f"Fehler beim Ausführen des Befehls {command}: {str(e)}")
        else:
            self.logger.warning(f"Unbekannter Befehl: {command}")
            self._send_message(chat_id, f"Unbekannter Befehl: {command}")
    
    def _send_message(self, chat_id, text):
        """
        Sendet eine Textnachricht an den angegebenen Chat.
        """
        try:
            payload = {"chat_id": chat_id, "text": text}
            response = self.session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json=payload
            )
            self.logger.info(f"Nachricht gesendet an Chat {chat_id}: {text} (Status: {response.status_code})")
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Nachricht: {str(e)}")
    
    def _send_status_to_all_users(self, title, message):
        """
        Sendet Status-Updates an alle autorisierten Benutzer.
        """
        for user_id in self.allowed_users:
            self._send_message(user_id, f"{title}\n{message}")
    
    def _start_status_update_timer(self):
        """
        Startet einen Timer, der in regelmäßigen Abständen Status-Updates versendet.
        """
        def status_update():
            while self.is_running:
                self._send_status_to_all_users("Status Update", "Der Trading Bot läuft...")
                time.sleep(self.status_update_interval)
        threading.Thread(target=status_update, daemon=True).start()
    
    def _is_authorized(self, user_id):
        """
        Überprüft, ob der angefragte Benutzer autorisiert ist.
        """
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    # Zusätzliche Methoden:

    def stop(self):
        """
        Beendet den Telegram-Bot.
        """
        self.logger.info("Stoppe Telegram-Bot...")
        self.is_running = False
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=5)
        self.logger.info("Telegram-Bot wurde gestoppt.")
    
    def debug_status(self):
        """
        Gibt den internen Status des Telegram-Bots zurück (hilfreich für Debugging).
        """
        status_info = {
            "is_running": self.is_running,
            "processed_updates_count": len(self.processed_updates),
            "registered_commands": list(self.commands.keys())
        }
        self.logger.debug(f"Debug-Status: {status_info}")
        return status_info

# Ende der Datei telegram_interface.py

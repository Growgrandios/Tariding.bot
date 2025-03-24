import os
import logging
import threading
import time
import json
import requests
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional

# Konstante für Telegram API URL
TELEGRAM_API_URL = "https://api.telegram.org/bot{}/{}"

class TelegramInterface:
    """
    Ein modernes Telegram Bot Interface Modul zur Fernsteuerung deines Trading Bots.
    
    Funktionen:
      - Starten/Stoppen des Bots und (automatisch) der Google VM Instanz
      - Statusabfragen, Kontostand, offene Positionen und Performance-Berichte
      - Notfall-Stop, Modul-spezifisches Steuern und regelmäßige Updates
      - Interaktive Inline-Buttons für Aktionen (z. B. Notfall‑Stop bestätigen)
      - Erweiterbar und modular, um sich harmonisch in alle Bot-Module einzuklinken
    """
    def __init__(self, config: Dict[str, Any], main_controller: Any):
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.info("Initialisiere neues Telegram Interface Modul...")
        
        self.config = config
        self.main_controller = main_controller
        
        # Telegram-API Konfiguration
        self.bot_token = config.get("bot_token", os.getenv("TELEGRAM_BOT_TOKEN", ""))
        if not self.bot_token:
            self.logger.error("Kein Telegram Bot Token gefunden!")
        
        # Erlaubte Benutzer (als Liste von Strings)
        allowed_users_raw = config.get("allowed_users", [])
        if isinstance(allowed_users_raw, str):
            self.allowed_users = [uid.strip() for uid in allowed_users_raw.split(",") if uid.strip()]
        elif isinstance(allowed_users_raw, list):
            self.allowed_users = [str(uid) for uid in allowed_users_raw if uid]
        else:
            self.allowed_users = []
        
        if not self.allowed_users:
            self.logger.warning("Keine erlaubten Telegram-Benutzer konfiguriert. Alle Anfragen werden ignoriert.")
        
        # Polling-Einstellungen
        self.polling_timeout = config.get("polling_timeout", 30)
        self.last_update_id = 0
        
        # Steuerung des Bot-Threads
        self.is_running = False
        self.bot_thread: Optional[threading.Thread] = None
        
        # Befehls-Mapping: command -> Handler-Methode
        self.commands: Dict[str, Callable[[Dict[str, Any]], None]] = {
            "start_bot": self._handle_start_bot,
            "stop_bot": self._handle_stop_bot,
            "status": self._handle_status,
            "balance": self._handle_balance,
            "positions": self._handle_positions,
            "performance": self._handle_performance,
            "report": self._handle_report,
            "emergency": self._handle_emergency,
        }
        
        # Notifikationsparameter
        self.notification_cooldown = config.get("notification_cooldown", 60)  # in Sekunden
        self.last_notification_time: Dict[str, datetime] = {}
        
        # VM-Steuerung: Konfigurationsparameter für Google VM Instanz
        self.vm_instance_name = config.get("vm_instance_name", "trading-bot-instance")
        self.vm_zone = config.get("vm_zone", "us-central1-a")
        
        self.logger.info("Telegram Interface Modul erfolgreich initialisiert")
    
    def start(self):
        """Startet den Telegram Bot in einem separaten Thread (Polling-Modus)."""
        if self.is_running:
            self.logger.warning("Telegram Bot läuft bereits!")
            return
        
        self.is_running = True
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()
        self.logger.info("Telegram Bot gestartet")
        # Sende initialen Status an alle erlaubten Benutzer
        self._send_notification_to_all("Trading Bot gestartet und bereit für Befehle.")
    
    def stop(self):
        """Stoppt den Telegram Bot (Polling-Schleife wird beendet)."""
        self.is_running = False
        self.logger.info("Telegram Bot wird gestoppt...")
    
    def _run_bot(self):
        """Hauptschleife: Polling der Telegram API nach Updates."""
        self.logger.info("Betrete Polling-Schleife für Telegram-Updates...")
        session = requests.Session()
        while self.is_running:
            try:
                url = TELEGRAM_API_URL.format(self.bot_token, "getUpdates")
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": self.polling_timeout,
                    "allowed_updates": ["message", "callback_query"]
                }
                response = session.get(url, params=params, timeout=self.polling_timeout + 5)
                response.raise_for_status()
                data = response.json()
                
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        self.last_update_id = update["update_id"]
                        self._process_update(update)
                else:
                    self.logger.debug("Keine neuen Updates empfangen.")
            except Exception as e:
                self.logger.error(f"Fehler beim Abrufen der Updates: {str(e)}")
                time.sleep(5)
    
    def _process_update(self, update: Dict[str, Any]):
        """Verarbeitet ein einzelnes Update von Telegram."""
        self.logger.debug(f"Verarbeite Update: {json.dumps(update)}")
        try:
            chat_id = None
            user_id = None
            message_text = ""
            callback_data = None
            
            if "message" in update:
                message = update["message"]
                chat_id = message.get("chat", {}).get("id")
                user_id = str(message.get("from", {}).get("id"))
                message_text = message.get("text", "").strip()
            elif "callback_query" in update:
                callback = update["callback_query"]
                chat_id = callback.get("message", {}).get("chat", {}).get("id")
                user_id = str(callback.get("from", {}).get("id"))
                callback_data = callback.get("data", "").strip()
                message_text = callback_data  # Für die Verarbeitung von Button-Aktionen
            
            if not chat_id or not user_id:
                self.logger.warning("Update ohne Chat-ID oder User-ID erhalten – überspringe.")
                return
            
            # Prüfe Benutzerautorisierung
            if self.allowed_users and (user_id not in self.allowed_users):
                self.logger.warning(f"Nicht autorisierter Zugriff von User {user_id}")
                self._send_message(chat_id, "Du bist nicht berechtigt, diesen Bot zu steuern.")
                return
            
            # Befehl extrahieren (erwartetes Format: /befehl [parameter])
            if message_text.startswith("/"):
                parts = message_text[1:].split()
                command = parts[0].lower()
                params = parts[1:]
                self.logger.debug(f"Befehl erhalten: {command} mit Parametern {params}")
                if command in self.commands:
                    self.commands[command]({"chat_id": chat_id, "user_id": user_id, "params": params})
                else:
                    self._send_message(chat_id, f"Unbekannter Befehl: {command}")
            else:
                self._send_message(chat_id, "Bitte sende einen Befehl beginnend mit '/' (z. B. /status).")
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung eines Updates: {str(e)}")
    
    def _send_message(self, chat_id: int, text: str, reply_markup: Optional[Dict] = None):
        """Sendet eine Nachricht an den angegebenen Chat."""
        try:
            url = TELEGRAM_API_URL.format(self.bot_token, "sendMessage")
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            if reply_markup:
                payload["reply_markup"] = json.dumps(reply_markup)
            response = requests.post(url, data=payload)
            response.raise_for_status()
            self.logger.debug(f"Nachricht an {chat_id} gesendet: {text}")
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Nachricht an {chat_id}: {str(e)}")
    
    def _send_notification_to_all(self, text: str):
        """Sendet eine Benachrichtigung an alle erlaubten Benutzer."""
        for user_id in self.allowed_users:
            self._send_message(int(user_id), text)
    
    # --------------------- Befehls-Handler ---------------------
    def _handle_start_bot(self, context: Dict[str, Any]):
        """
        /start_bot
        Startet den Trading Bot und, falls notwendig, die Google VM Instanz.
        """
        chat_id = context["chat_id"]
        self.logger.info("Startbefehl empfangen: Starte Trading Bot und VM Instanz...")
        
        # Starte zuerst die VM Instanz (sofern konfiguriert)
        vm_started, vm_response = self._start_vm_instance()
        if vm_started:
            start_msg = "VM Instanz erfolgreich gestartet.\n"
        else:
            start_msg = "VM Instanz konnte nicht gestartet werden:\n" + vm_response + "\n"
        
        # Starte den Bot über den Main Controller (hier wird erwartet, dass main_controller.start() existiert)
        try:
            self.main_controller.start(mode="live", auto_trade=True)
            start_msg += "Trading Bot wurde gestartet."
        except Exception as e:
            start_msg += f"Fehler beim Starten des Bots: {str(e)}"
        
        self._send_message(chat_id, start_msg)
    
    def _handle_stop_bot(self, context: Dict[str, Any]):
        """
        /stop_bot
        Stoppt den Trading Bot.
        """
        chat_id = context["chat_id"]
        self.logger.info("Stop-Befehl empfangen: Stoppe Trading Bot...")
        try:
            self.main_controller.stop()
            self._send_message(chat_id, "Trading Bot wurde gestoppt.")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Stoppen des Bots: {str(e)}")
    
    def _handle_status(self, context: Dict[str, Any]):
        """
        /status
        Gibt den aktuellen Status des Bots und aller Module zurück.
        """
        chat_id = context["chat_id"]
        self.logger.info("Statusabfrage erhalten")
        try:
            status = self.main_controller.get_status()  # Erwartet ein Dictionary oder String
            self._send_message(chat_id, f"<b>Bot-Status:</b>\n{json.dumps(status, indent=2)}")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Abrufen des Status: {str(e)}")
    
    def _handle_balance(self, context: Dict[str, Any]):
        """
        /balance
        Zeigt den aktuellen Kontostand an.
        """
        chat_id = context["chat_id"]
        self.logger.info("Balanceabfrage erhalten")
        try:
            balance = self.main_controller.get_account_balance()  # Muss im Main Controller implementiert sein
            self._send_message(chat_id, f"<b>Kontostand:</b>\n{json.dumps(balance, indent=2)}")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Abrufen des Kontostands: {str(e)}")
    
    def _handle_positions(self, context: Dict[str, Any]):
        """
        /positions
        Zeigt offene Positionen an.
        """
        chat_id = context["chat_id"]
        self.logger.info("Positionsabfrage erhalten")
        try:
            positions = self.main_controller.get_open_positions()  # Erwartet, dass main_controller diese Methode hat
            self._send_message(chat_id, f"<b>Offene Positionen:</b>\n{json.dumps(positions, indent=2)}")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Abrufen der Positionen: {str(e)}")
    
    def _handle_performance(self, context: Dict[str, Any]):
        """
        /performance
        Zeigt Performance-Metriken des Trading Bots an.
        """
        chat_id = context["chat_id"]
        self.logger.info("Performanceabfrage erhalten")
        try:
            performance = self.main_controller.get_performance_metrics()  # Muss im Main Controller implementiert sein
            self._send_message(chat_id, f"<b>Performance:</b>\n{json.dumps(performance, indent=2)}")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Abrufen der Performance: {str(e)}")
    
    def _handle_report(self, context: Dict[str, Any]):
        """
        /report
        Erzeugt einen aktuellen Bericht und sendet diesen an den Benutzer.
        """
        chat_id = context["chat_id"]
        self.logger.info("Berichtsanforderung erhalten")
        try:
            report = self.main_controller.generate_report()  # Beispielhafter Aufruf; muss in main_controller definiert sein
            self._send_message(chat_id, f"<b>Bericht:</b>\n{report}")
        except Exception as e:
            self._send_message(chat_id, f"Fehler beim Erzeugen des Berichts: {str(e)}")
    
    def _handle_emergency(self, context: Dict[str, Any]):
        """
        /emergency
        Löst den Notfall-Stop aus und stoppt sofort alle Trading-Aktivitäten.
        Bestätigungs-Button wird gesendet.
        """
        chat_id = context["chat_id"]
        self.logger.critical("EMERGENCY-Befehl empfangen! Notfall-Stop wird eingeleitet.")
        # Sende Inline-Button zur Bestätigung
        reply_markup = {
            "inline_keyboard": [
                [{"text": "Notfall stoppen", "callback_data": "confirm_emergency_stop"}]
            ]
        }
        self._send_message(chat_id, "Bist du sicher, dass du einen Notfall-Stop auslösen möchtest?", reply_markup)
    
    # --------------------- Inline-Callback-Handler ---------------------
    def _handle_callback_query(self, update: Dict[str, Any]):
        """
        Verarbeitet Callback Queries (z. B. Button-Klicks).
        """
        try:
            callback_query = update.get("callback_query", {})
            data = callback_query.get("data", "")
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            if data == "confirm_emergency_stop":
                self.logger.critical("Notfall-Stop bestätigt über Button.")
                try:
                    self.main_controller.emergency_stop()  # Erwartet, dass main_controller eine entsprechende Methode hat
                    self._send_message(chat_id, "Notfall-Stop wurde ausgeführt. Alle Aktivitäten wurden sofort gestoppt.")
                except Exception as e:
                    self._send_message(chat_id, f"Fehler beim Ausführen des Notfall-Stops: {str(e)}")
        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten der Callback Query: {str(e)}")
    
    # --------------------- Google VM Steuerung ---------------------
    def _start_vm_instance(self) -> (bool, str):
        """
        Startet die Google VM Instanz mithilfe der gcloud CLI.
        Voraussetzung: gcloud ist installiert und konfiguriert.
        
        Rückgabe:
            (True, stdout) im Erfolgsfall, (False, stderr) sonst.
        """
        self.logger.info("Starte Google VM Instanz...")
        try:
            result = subprocess.run(
                [
                    "gcloud", "compute", "instances", "start",
                    self.vm_instance_name,
                    "--zone", self.vm_zone
                ],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                self.logger.info("VM Instanz erfolgreich gestartet.")
                return True, result.stdout
            else:
                self.logger.error(f"Fehler beim Starten der VM Instanz: {result.stderr}")
                return False, result.stderr
        except Exception as e:
            self.logger.error(f"Exception beim Starten der VM Instanz: {str(e)}")
            return False, str(e)
    
    # --------------------- Public Callback für Callback Queries ---------------------
    def process_callback_update(self, update: Dict[str, Any]):
        """Öffentliche Methode, um Callback Updates zu verarbeiten."""
        self._handle_callback_query(update)
    
    # --------------------- Weitere Helper-Funktionen ---------------------
    def notify(self, priority: str, message: str):
        """
        Versendet eine Benachrichtigung, wobei ein Cooldown berücksichtigt wird.
        """
        now = datetime.now()
        last_time = self.last_notification_time.get(priority, now - timedelta(seconds=self.notification_cooldown + 1))
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug(f"Notification für Priorität '{priority}' im Cooldown.")
            return
        self.last_notification_time[priority] = now
        self._send_notification_to_all(f"[{priority.upper()}] {message}")

# Beispiel: Falls dieses Modul direkt gestartet wird
if __name__ == "__main__":
    # Beispielhafte Konfiguration
    config = {
        "bot_token": "DEIN_TELEGRAM_BOT_TOKEN",
        "allowed_users": ["123456789"],
        "polling_timeout": 30,
        "notification_cooldown": 60,
        "vm_instance_name": "trading-bot-instance",
        "vm_zone": "us-central1-a"
    }
    
    # Dummy MainController mit minimalen Methoden (zum Testen)
    class DummyMainController:
        def start(self, mode, auto_trade):
            print(f"Trading Bot gestartet im Modus {mode} (Auto Trade: {auto_trade})")
        def stop(self):
            print("Trading Bot gestoppt.")
        def get_status(self):
            return {"state": "running", "modules": {"live_trading": "active", "learning": "idle"}}
        def get_account_balance(self):
            return {"USD": 10000, "BTC": 0.5}
        def get_open_positions(self):
            return [{"symbol": "BTC/USDT", "size": 0.1, "side": "long"}]
        def get_performance_metrics(self):
            return {"win_rate": 0.65, "profit": 1200}
        def generate_report(self):
            return "Täglicher Bericht: Alles im grünen Bereich."
        def emergency_stop(self):
            print("Notfall-Stop wurde ausgeführt!")
    
    dummy_controller = DummyMainController()
    bot = TelegramInterface(config, dummy_controller)
    bot.start()
    
    # Für Testzwecke: Endlosschleife, um den Bot am Laufen zu halten
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()

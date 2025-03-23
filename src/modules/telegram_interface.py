# telegram_interface.py

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
    """
    def __init__(self, config: Dict[str, Any], main_controller=None):
        """
        Initialisiert die Telegram-Schnittstelle.
        """
        # Konsolen-Handler für direkte Ausgabe einrichten
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('\033[92m%(asctime)s - TELEGRAM - %(levelname)s - %(message)s\033[0m'))
        
        # Logger mit dem Console-Handler konfigurieren
        self.logger = logging.getLogger("TelegramInterface")
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
        print("\n" + "="*80)
        print("TELEGRAM INTERFACE WIRD INITIALISIERT")
        print("="*80 + "\n")
        
        self.logger.info("Initialisiere TelegramInterface...")

        # Debug-Modus aktivieren - WICHTIG für Fehlerbehebung
        self.debug_mode = config.get('debug_mode', True)
        if self.debug_mode:
            self.logger.info("Debug-Modus aktiviert - alle Button-Aktionen werden protokolliert")

        # API-Konfiguration
        self.bot_token = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
        print(f"Bot-Token: {'*'*5}{self.bot_token[-5:] if self.bot_token else 'NICHT KONFIGURIERT'}")
        
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
            print("KRITISCHER FEHLER: Kein Telegram-Bot-Token konfiguriert!")
            self.is_configured = False
        elif not self.allowed_users:
            self.logger.warning("Keine erlaubten Telegram-Benutzer konfiguriert")
            print("WARNUNG: Keine erlaubten Telegram-Benutzer konfiguriert!")
            self.is_configured = True  # Wir können trotzdem starten, aber keine Befehle annehmen
        else:
            self.is_configured = True
            print(f"Erlaubte Benutzer: {self.allowed_users}")
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
        print("\n" + "="*80)
        print("TELEGRAM INTERFACE BEREIT")
        print("="*80 + "\n")

    def register_commands(self, commands: Dict[str, Callable]):
        """
        Registriert benutzerdefinierte Befehle vom MainController.
        """
        print(f"Registriere {len(commands)} benutzerdefinierte Befehle:")
        for cmd in commands:
            print(f" - /{cmd}")
            
        self.logger.info(f"Registriere {len(commands)} benutzerdefinierte Befehle")
        self.commands = commands

    def start(self):
        """Startet den Telegram-Bot in einem separaten Thread."""
        if not self.is_configured:
            self.logger.warning("Telegram-Bot nicht konfiguriert, kann nicht gestartet werden")
            print("FEHLER: Telegram-Bot nicht konfiguriert, kann nicht gestartet werden")
            return False

        if self.is_running:
            self.logger.warning("Telegram-Bot läuft bereits")
            return True

        try:
            print("\n" + "="*80)
            print("STARTE TELEGRAM BOT...")
            print("="*80 + "\n")
            
            self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
            self.bot_thread.start()
            
            # Kurz warten, um sicherzustellen, dass der Bot gestartet wird
            time.sleep(1)
            self.is_running = True
            self.logger.info("Telegram-Bot gestartet")
            print("Telegram-Bot erfolgreich gestartet! Warte auf Befehle...")

            # Initialen Status an alle Benutzer senden
            self._send_status_to_all_users("Bot gestartet", "Der Trading Bot wurde erfolgreich gestartet und ist bereit für Befehle.")

            # Timer für regelmäßige Statusupdates starten
            if self.status_update_interval > 0:
                self._start_status_update_timer()

            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Telegram-Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            print(f"KRITISCHER FEHLER beim Starten des Telegram-Bots: {str(e)}")
            return False

    def _run_bot(self):
        """Führt den Telegram-Bot komplett ohne asyncio und Signal-Handler aus"""
        try:
            self.session = requests.Session()
            last_update_id = 0
            self.logger.info("Bot-Thread gestartet (HTTP-Polling-Modus)")
            print("Bot-Thread gestartet - HTTP-Polling aktiv")

            while self.is_running:
                try:
                    # Direkter API-Aufruf mit Long-Polling
                    print(f"Polling für Updates (offset={last_update_id + 1})...")
                    response = self.session.get(
                        f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                        params={
                            "offset": last_update_id + 1,
                            "timeout": 30,
                            "allowed_updates": ["message", "callback_query"]
                        },
                        timeout=35
                    )
                    
                    if response.status_code != 200:
                        print(f"FEHLER: API-Anfrage fehlgeschlagen mit Status {response.status_code}")
                        print(f"Antwort: {response.text}")
                        self.logger.error(f"API-Anfrage fehlgeschlagen: {response.status_code} - {response.text}")
                        time.sleep(5)
                        continue

                    response.raise_for_status()
                    
                    # Verarbeite Updates
                    data = response.json()
                    if not data.get("ok", False):
                        print(f"FEHLER: API-Anfrage nicht erfolgreich: {data.get('description', 'Unbekannter Fehler')}")
                        self.logger.error(f"API-Anfrage nicht erfolgreich: {data}")
                        time.sleep(5)
                        continue
                        
                    updates = data.get("result", [])
                    if updates:
                        print(f"EMPFANGEN: {len(updates)} neue Updates")
                    
                    for update in updates:
                        last_update_id = update["update_id"]
                        
                        # Prüfen, ob dieses Update bereits verarbeitet wurde
                        if update["update_id"] not in self.processed_updates:
                            self._handle_raw_update(update)
                            self.processed_updates.add(update["update_id"])
                        else:
                            print(f"Update {update['update_id']} wurde bereits verarbeitet, überspringe")
                    
                    # Begrenze die Größe der verarbeiteten Updates
                    if len(self.processed_updates) > 1000:
                        self.processed_updates = set(list(self.processed_updates)[-500:])
                        
                except Exception as e:
                    self.logger.error(f"Polling-Fehler: {str(e)}")
                    print(f"POLLING-FEHLER: {str(e)}")
                    traceback.print_exc()
                    time.sleep(5)

        except Exception as e:
            self.logger.error(f"Kritischer Bot-Fehler: {str(e)}")
            self.logger.error(traceback.format_exc())
            print(f"KRITISCHER BOT-FEHLER: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.info("Bot-Thread beendet")
            print("Bot-Thread beendet")
            self.is_running = False

    def _handle_raw_update(self, update):
        """Verarbeitet ein Telegram-Update mit ausführlichem Debugging"""
        try:
            # Ausführliches Debugging - Zeige das vollständige Update in der Konsole
            print("\n" + "="*80)
            update_type = "BUTTON-DRUCK" if "callback_query" in update else "NACHRICHT"
            print(f"TELEGRAM {update_type} EMPFANGEN (ID: {update.get('update_id')})")
            print("-"*80)
            print(json.dumps(update, indent=2))
            print("="*80 + "\n")
            
            self.logger.info(f"Update-ID: {update.get('update_id')} - Typ: {'callback_query' if 'callback_query' in update else 'message'}")

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
                
                print(f"NACHRICHT: Text='{text}', Chat-ID={chat_id}, User-ID={user_id}, Message-ID={message_id}")
                
            elif "callback_query" in update:
                self.logger.info("Callback-Query (Button-Druck) erkannt")
                callback_query = update["callback_query"]
                chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
                user_id = callback_query.get("from", {}).get("id")
                callback_data = callback_query.get("data")
                message_id = callback_query.get("message", {}).get("message_id")
                
                print(f"BUTTON GEDRÜCKT: Data='{callback_data}', Chat-ID={chat_id}, User-ID={user_id}, Message-ID={message_id}")
                self.logger.info(f"BUTTON GEDRÜCKT: {callback_data}")

            # Autorisierung prüfen
            if not chat_id:
                self.logger.warning(f"Keine Chat-ID gefunden in Update {update.get('update_id')}")
                print(f"FEHLER: Keine Chat-ID gefunden in Update {update.get('update_id')}")
                return
                
            if not self._is_authorized(str(user_id)):
                self.logger.warning(f"Nicht autorisierter Zugriff von User ID: {user_id}")
                print(f"SICHERHEIT: Nicht autorisierter Zugriff von User ID: {user_id}")
                return

            # Callback-Query beantworten (WICHTIG für Buttons)
            if "callback_query" in update:
                try:
                    callback_id = update["callback_query"]["id"]
                    print(f"Beantworte Callback-ID: {callback_id}")
                    self.logger.info(f"Beantworte Callback: {callback_id}")

                    # Antwort an Telegram senden (wichtig!)
                    answer_url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
                    answer_data = {"callback_query_id": callback_id}
                    
                    print(f"API-Anfrage: POST {answer_url} mit Daten: {answer_data}")
                    response = self.session.post(answer_url, json=answer_data)
                    
                    print(f"TELEGRAM ANTWORT: Status={response.status_code}")
                    print(f"Antwort-Inhalt: {response.text}")
                    
                    self.logger.info(f"Antwort von Telegram: {response.status_code}, {response.text}")
                    
                    # Kurze Verzögerung für bessere Verarbeitung
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"CALLBACK-FEHLER: {str(e)}")
                    traceback.print_exc()
                    self.logger.error(f"Fehler beim Beantworten der Callback-Query: {str(e)}")

            # Callback-Anfragen verarbeiten
            if callback_data:
                self.logger.info(f"Verarbeite Callback-Daten: {callback_data}")
                print(f"Verarbeite Button-Callback: '{callback_data}' von Chat {chat_id}")
                self._handle_callback_data(chat_id, callback_data, message_id)
                return

            # Textnachrichten verarbeiten
            if text:
                self.logger.info(f"Verarbeite Text: {text}")
                print(f"Verarbeite Text: '{text}' von Chat {chat_id}")
                
                # Befehle verarbeiten (beginnen mit /)
                if text.startswith("/"):
                    self._process_command(chat_id, text)
                # Normale Nachrichten
                else:
                    self._send_direct_message(chat_id, "Ich verstehe nur Befehle. Verwende /help für eine Liste der verfügbaren Befehle.")

        except Exception as e:
            self.logger.error(f"Fehler bei der Update-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            print(f"FEHLER bei Update-Verarbeitung: {str(e)}")
            traceback.print_exc()
    def _handle_callback_data(self, chat_id, callback_data, message_id=None):
        """Verarbeitet Callback-Daten von Inline-Buttons"""
        try:
            self.logger.info(f"Verarbeite Callback-Daten: {callback_data} von Chat {chat_id}")
            print(f"\n{'='*40}")
            print(f"BUTTON-AKTION: '{callback_data}' von Chat {chat_id}")
            print(f"{'='*40}\n")

            # Debugging: Callback ins Log schreiben
            with open("button_log.txt", "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - Chat {chat_id} - Button: {callback_data}\n")

            # Je nach Callback-Data unterschiedliche Aktionen ausführen
            if callback_data == "startbot":
                print(f"Aktion: Bot starten")
                self._handle_start_bot(chat_id)
            elif callback_data == "stopbot":
                print(f"Aktion: Bot stoppen")
                self._handle_stop_bot(chat_id)
            elif callback_data == "pausebot":
                print(f"Aktion: Bot pausieren")
                self._handle_pause_bot(chat_id)
            elif callback_data == "resumebot":
                print(f"Aktion: Bot fortsetzen")
                self._handle_resume_bot(chat_id)
            elif callback_data == "balance":
                print(f"Aktion: Kontostand abrufen")
                self._handle_balance(chat_id)
            elif callback_data == "positions":
                print(f"Aktion: Positionen abrufen")
                self._handle_positions(chat_id)
            elif callback_data == "performance":
                print(f"Aktion: Performance abrufen")
                self._handle_performance(chat_id)
            elif callback_data == "dashboard":
                print(f"Aktion: Dashboard anzeigen")
                self._send_dashboard(chat_id)
            elif callback_data == "refresh_status":
                print(f"Aktion: Status aktualisieren")
                self._send_status_message(chat_id, message_id)
            elif callback_data == "help":
                print(f"Aktion: Hilfe anzeigen")
                self._process_command(chat_id, "/help")
            elif callback_data == "status":
                print(f"Aktion: Status anzeigen")
                self._send_status_message(chat_id)
            elif callback_data == "refresh_positions":
                print(f"Aktion: Positionen aktualisieren")
                self._handle_positions(chat_id)
            elif callback_data == "close_all_positions":
                print(f"Aktion: Alle Positionen schließen (Bestätigung)")
                self._confirm_close_all_positions(chat_id)
            elif callback_data == "confirm_close_all":
                print(f"Aktion: Alle Positionen schließen (Ausführung)")
                self._execute_close_all_positions(chat_id)
            elif callback_data == "cancel_close_all":
                print(f"Aktion: Schließen aller Positionen abgebrochen")
                self._send_direct_message(chat_id, "Abgebrochen. Keine Positionen wurden geschlossen.")
            elif callback_data.startswith("train_model:"):
                parts = callback_data.split(":")
                if len(parts) >= 3:
                    symbol = parts[1]
                    timeframe = parts[2]
                    print(f"Aktion: Modell trainieren für {symbol} ({timeframe})")
                    self._handle_train_model(chat_id, symbol, timeframe)
            elif callback_data.startswith("analyze_news:"):
                asset = callback_data.split(":")[1]
                print(f"Aktion: News analysieren für {asset}")
                self._handle_analyze_news(chat_id, asset)
            elif callback_data.startswith("optimize_portfolio"):
                print(f"Aktion: Portfolio optimieren")
                self._handle_optimize_portfolio(chat_id)
            else:
                print(f"Aktion: Unbekannte Callback-Daten: {callback_data}")
                self._send_direct_message(chat_id, f"Unbekannte Aktion: {callback_data}")

        except Exception as e:
            self.logger.error(f"Fehler bei der Callback-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            print(f"FEHLER bei Button-Verarbeitung: {str(e)}")
            traceback.print_exc()
            self._send_direct_message(chat_id, f"Fehler bei der Ausführung: {str(e)}")

    def _process_command(self, chat_id, command_text):
        """Verarbeitet einen Befehl"""
        # Befehl und Parameter extrahieren
        parts = command_text.split(maxsplit=1)
        command = parts[0][1:]  # Entferne das '/'
        params = parts[1] if len(parts) > 1 else ""

        print(f"\n{'='*40}")
        print(f"BEFEHL: /{command} {params} von Chat {chat_id}")
        print(f"{'='*40}\n")
        
        with open("command_log.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Chat {chat_id} - Befehl: /{command} {params}\n")

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
            print("Befehl: Start - Zeige Willkommensnachricht")
            self._send_direct_message(chat_id, "🤖 Bot aktiv! Nutze /help für Befehle")
        elif command == "help":
            print("Befehl: Help - Zeige Hilfetext")
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
            print("Befehl: Status - Zeige aktuellen Botstatus")
            if self.main_controller:
                self._send_status_message(chat_id)
            else:
                self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
        elif command == "startbot":
            print("Befehl: StartBot - Starte Trading Bot")
            self._handle_start_bot(chat_id)
        elif command == "stopbot":
            print("Befehl: StopBot - Stoppe Trading Bot")
            self._handle_stop_bot(chat_id)
        elif command == "pausebot":
            print("Befehl: PauseBot - Pausiere Trading Bot")
            self._handle_pause_bot(chat_id)
        elif command == "resumebot":
            print("Befehl: ResumeBot - Setze Trading Bot fort")
            self._handle_resume_bot(chat_id)
        elif command == "balance":
            print("Befehl: Balance - Zeige Kontostand")
            self._handle_balance(chat_id)
        elif command == "positions":
            print("Befehl: Positions - Zeige offene Positionen")
            self._handle_positions(chat_id)
        elif command == "performance":
            print("Befehl: Performance - Zeige Performance-Metriken")
            self._handle_performance(chat_id)
        elif command == "dashboard":
            print("Befehl: Dashboard - Zeige Dashboard")
            self._send_dashboard(chat_id)
        elif command == "processtranscript":
            print(f"Befehl: ProcessTranscript - Verarbeite Transkript: {params}")
            if params:
                self._handle_process_transcript(chat_id, params)
            else:
                self._send_direct_message(chat_id, "Bitte gib den Pfad zum Transkript an.\nBeispiel: /processtranscript data/transcripts/mein_transkript.txt")
        else:
            # Prüfen, ob ein benutzerdefinierter Befehl registriert ist
            if command in self.commands:
                try:
                    print(f"Befehl: {command} (benutzerdefiniert) - Führe aus mit Parametern: {params}")
                    result = self.commands[command]({"chat_id": chat_id, "params": params})
                    response = result.get("message", f"Befehl '{command}' ausgeführt")
                    self._send_direct_message(chat_id, response)
                except Exception as e:
                    print(f"FEHLER bei benutzerdefiniertem Befehl '{command}': {str(e)}")
                    self._send_direct_message(chat_id, f"Fehler beim Ausführen des Befehls '{command}': {str(e)}")
            else:
                print(f"Befehl: Unbekannter Befehl: {command}")
                self._send_direct_message(chat_id, f"Unbekannter Befehl: /{command}\nVerwende /help für verfügbare Befehle.")

    def _is_authorized(self, user_id):
        """Prüft, ob ein Benutzer autorisiert ist"""
        if not self.allowed_users:
            self.logger.warning("Keine erlaubten Benutzer konfiguriert")
            return False
            
        is_authorized = str(user_id) in self.allowed_users
        
        if is_authorized:
            print(f"Benutzer {user_id} ist autorisiert ✓")
        else:
            print(f"Benutzer {user_id} ist NICHT autorisiert ✗")
            
        return is_authorized

    def _send_direct_message(self, chat_id, text, reply_markup=None, parse_mode=None):
        """Sendet eine Direktnachricht an einen Chat"""
        if not self.is_running or not self.is_configured:
            self.logger.warning(f"Bot nicht bereit zum Senden von Nachrichten an {chat_id}")
            return False

        try:
            print(f"Sende Nachricht an Chat {chat_id}: {text[:50]}...")
            
            data = {
                "chat_id": chat_id,
                "text": text
            }

            if parse_mode:
                data["parse_mode"] = parse_mode

            if reply_markup:
                data["reply_markup"] = reply_markup

            response = self.session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json=data
            )
            
            if response.status_code != 200:
                print(f"FEHLER beim Senden der Nachricht: {response.status_code}")
                print(response.text)
                self.logger.error(f"Fehler beim Senden der Nachricht: {response.status_code} - {response.text}")
                return False
                
            print(f"Nachricht erfolgreich gesendet: Status {response.status_code}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Nachricht: {str(e)}")
            print(f"FEHLER beim Senden der Nachricht: {str(e)}")
            return False

    def _send_photo(self, chat_id, photo_path, caption=None, reply_markup=None):
        """Sendet ein Foto an einen Chat"""
        if not self.is_running or not self.is_configured:
            self.logger.warning(f"Bot nicht bereit zum Senden von Fotos an {chat_id}")
            return False

        try:
            print(f"Sende Foto {photo_path} an Chat {chat_id}")
            
            with open(photo_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {'chat_id': chat_id}
                
                if caption:
                    data['caption'] = caption
                    
                if reply_markup:
                    data['reply_markup'] = json.dumps(reply_markup)

                response = self.session.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendPhoto",
                    data=data,
                    files=files
                )
                
                if response.status_code != 200:
                    print(f"FEHLER beim Senden des Fotos: {response.status_code}")
                    print(response.text)
                    self.logger.error(f"Fehler beim Senden des Fotos: {response.status_code} - {response.text}")
                    return False
                    
                print(f"Foto erfolgreich gesendet: Status {response.status_code}")
                return True
        except Exception as e:
            self.logger.error(f"Fehler beim Senden des Fotos: {str(e)}")
            print(f"FEHLER beim Senden des Fotos: {str(e)}")
            traceback.print_exc()
            return False

    def _handle_start_bot(self, chat_id):
        """Startet den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return

        print("Aktion: Trading Bot starten")
        self._send_direct_message(chat_id, "Starte Trading Bot...")

        try:
            result = self.main_controller.start()
            
            print(f"Ergebnis des Bot-Starts: {result}")
            
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich gestartet")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Starten des Trading Bots")
        except Exception as e:
            print(f"FEHLER beim Bot-Start: {str(e)}")
            traceback.print_exc()
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")

    def _handle_stop_bot(self, chat_id):
        """Stoppt den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return

        print("Aktion: Trading Bot stoppen")
        self._send_direct_message(chat_id, "Stoppe Trading Bot...")

        try:
            result = self.main_controller.stop()
            
            print(f"Ergebnis des Bot-Stopps: {result}")
            
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich gestoppt")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Stoppen des Trading Bots")
        except Exception as e:
            print(f"FEHLER beim Bot-Stopp: {str(e)}")
            traceback.print_exc()
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")

    def _handle_pause_bot(self, chat_id):
        """Pausiert den Trading Bot"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return

        print("Aktion: Trading Bot pausieren")
        self._send_direct_message(chat_id, "Pausiere Trading Bot...")

        try:
            result = self.main_controller.pause()
            
            print(f"Ergebnis der Bot-Pause: {result}")
            
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich pausiert")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Pausieren des Trading Bots")
        except Exception as e:
            print(f"FEHLER beim Bot-Pausieren: {str(e)}")
            traceback.print_exc()
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")

    def _handle_resume_bot(self, chat_id):
        """Setzt den pausierten Trading Bot fort"""
        if not self.main_controller:
            self._send_direct_message(chat_id, "Fehler: Kein Zugriff auf den MainController")
            return

        print("Aktion: Trading Bot fortsetzen")
        self._send_direct_message(chat_id, "Setze Trading Bot fort...")

        try:
            result = self.main_controller.resume()
            
            print(f"Ergebnis der Bot-Fortsetzung: {result}")
            
            if result:
                self._send_direct_message(chat_id, "✅ Trading Bot erfolgreich fortgesetzt")
            else:
                self._send_direct_message(chat_id, "❌ Fehler beim Fortsetzen des Trading Bots")
        except Exception as e:
            print(f"FEHLER beim Bot-Fortsetzen: {str(e)}")
            traceback.print_exc()
            self._send_direct_message(chat_id, f"❌ Fehler: {str(e)}")

    # Restliche Methoden wie _handle_balance, _handle_positions usw. würden hier folgen
    # mit ähnlichen Debug-Verbesserungen

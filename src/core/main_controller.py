# main_controller.py

import os
import sys
import logging
import threading
import time
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import traceback
import signal
import importlib
import queue
from concurrent.futures import ThreadPoolExecutor

# Module importieren
from src.core.config_manager import ConfigManager
from src.modules.data_pipeline import DataPipeline
from src.modules.live_trading import LiveTradingConnector
from src.modules.learning_module import LearningModule
from src.modules.transcript_processor import TranscriptProcessor
from src.modules.black_swan_detector import BlackSwanDetector
from src.modules.telegram_interface import TelegramInterface
from src.modules.tax_module import TaxModule

# Logger einrichten (ohne basicConfig, da dies bereits in main.py erfolgt)
logger = logging.getLogger("MainController")

class BotState:
    """Status des Trading Bots"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class MainController:
    """
    Hauptcontroller für den Trading Bot.
    Koordiniert und verwaltet alle Module und steuert den Gesamtbetrieb.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialisiert den MainController.
        
        Args:
            config_manager: Ein ConfigManager-Objekt für die Konfigurationsverwaltung (optional)
        """
        self.logger = logging.getLogger("MainController")
        self.logger.info("Initialisiere MainController...")
        
        # Verwende den übergebenen ConfigManager
        self.config_manager = config_manager
        
        # Bot-Status
        self.state = BotState.INITIALIZING
        self.previous_state = None
        self.emergency_mode = False
        
        # Ereignisprotokollierung
        self.events = []
        self.max_events = 1000  # Maximale Anzahl der gespeicherten Ereignisse
        
        # Steuerungs-Flags
        self.running = False
        self.shutdown_requested = False
        self.restart_requested = False
        self.pause_requested = False
        
        # Threads
        self.main_thread = None
        self.monitor_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Event-Queue für asynchrone Kommunikation
        self.event_queue = queue.Queue()
        
        # Signal-Handler für Graceful Shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Module
        self.modules = {}
        self.module_status = {}
        
        # Konfiguration laden
        if self.config_manager:
            self.config = self.config_manager.get_config()
            
            # Log-Level anpassen
            log_level = self.config.get('general', {}).get('log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, log_level))
            
            # Pfade
            self.data_path = Path(self.config.get('general', {}).get('data_path', 'data'))
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Module initialisieren
            self._initialize_modules()
            
            # Status auf bereit setzen
            self.state = BotState.READY
            self.logger.info("MainController erfolgreich initialisiert")
        else:
            self.logger.error("Kein ConfigManager übergeben, MainController nicht vollständig initialisiert")
            self.config = {}
            self.data_path = Path('data')
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.state = BotState.ERROR
    
    def _signal_handler(self, sig, frame):
        """Behandelt Betriebssystem-Signale für sauberes Herunterfahren."""
        self.logger.info(f"Signal {sig} empfangen. Fahre Bot herunter...")
        self.shutdown_requested = True
        if self.state == BotState.RUNNING:
            self.stop()
    
    def _initialize_modules(self):
        """Initialisiert alle Module des Trading Bots."""
        try:
            # Dictionary zum Speichern der Module
            self.modules = {}
            self.module_status = {}
            
            # Datenpipeline
            self.logger.info("Initialisiere DataPipeline...")
            data_config = self.config_manager.get_config('data_pipeline')
            api_keys = self.config_manager.get_api_keys()
            self.data_pipeline = DataPipeline(api_keys)
            self.modules['data_pipeline'] = self.data_pipeline
            self.module_status['data_pipeline'] = {"status": "initialized", "errors": []}
            
            # Black Swan Detector
            self.logger.info("Initialisiere BlackSwanDetector...")
            blackswan_config = self.config_manager.get_config('black_swan_detector')
            self.black_swan_detector = BlackSwanDetector(blackswan_config)
            self.modules['black_swan_detector'] = self.black_swan_detector
            self.module_status['black_swan_detector'] = {"status": "initialized", "errors": []}
            
            # Verbinde BlackSwanDetector mit DataPipeline
            self.black_swan_detector.set_data_pipeline(self.data_pipeline)
            
            # Live Trading Connector
            self.logger.info("Initialisiere LiveTradingConnector...")
            trading_config = self.config_manager.get_config('trading')
            # API-Schlüssel für Bitget hinzufügen
            trading_config.update(self.config_manager.get_api_key('bitget'))
            self.live_trading = LiveTradingConnector(trading_config)
            self.modules['live_trading'] = self.live_trading
            self.module_status['live_trading'] = {"status": "initialized", "errors": []}
            
            # Learning Module
            self.logger.info("Initialisiere LearningModule...")
            learning_config = self.config_manager.get_config('learning_module')
            self.learning_module = LearningModule(learning_config)
            self.modules['learning_module'] = self.learning_module
            self.module_status['learning_module'] = {"status": "initialized", "errors": []}
            
            # Telegram Interface
            self.logger.info("Initialisiere TelegramInterface...")
            telegram_config = self.config_manager.get_config('telegram')
            # Bot-Token und erlaubte Benutzer hinzufügen
            telegram_config.update(self.config_manager.get_api_key('telegram'))
            self.telegram_interface = TelegramInterface(telegram_config, self)
            self.modules['telegram_interface'] = self.telegram_interface
            self.module_status['telegram_interface'] = {"status": "initialized", "errors": []}
            
            # Transcript Processor
            self.logger.info("Initialisiere TranscriptProcessor...")
            transcript_config = self.config_manager.get_config('transcript_processor') or {}
            self.transcript_processor = TranscriptProcessor(transcript_config)
            self.modules['transcript_processor'] = self.transcript_processor
            self.module_status['transcript_processor'] = {"status": "initialized", "errors": []}
            
            # Tax Module
            self.logger.info("Initialisiere TaxModule...")
            tax_config = self.config_manager.get_config('tax_module')
            self.tax_module = TaxModule(tax_config)
            self.modules['tax_module'] = self.tax_module
            self.module_status['tax_module'] = {"status": "initialized", "errors": []}
            
            # Module miteinander verbinden
            self._connect_modules()
            
            self.logger.info("Alle Module erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise
            
    def _connect_modules(self):
        """Verbindet die Module miteinander für Kommunikation und Datenaustausch."""
        try:
            # Black Swan Detector mit Live Trading verbinden
            self.black_swan_detector.register_notification_callback(self._handle_black_swan_event)
            
            # Telegram Interface Callbacks registrieren
            telegram_commands = {
                'start': self.start,
                'stop': self.stop,
                'status': self.get_status,
                'balance': self._get_account_balance,
                'positions': self._get_open_positions,
                'performance': self._get_performance_metrics,
                'process_transcript': self._process_transcript_command
            }
            self.telegram_interface.register_commands(telegram_commands)
            
            # Live Trading Error-Callbacks registrieren
            self.live_trading.register_error_callback(self._handle_trading_error)
            self.live_trading.register_order_update_callback(self._handle_order_update)
            self.live_trading.register_position_update_callback(self._handle_position_update)
            
            # Tax Module mit Live Trading verbinden
            self.live_trading.register_order_update_callback(self.tax_module.process_trade)
            
            self.logger.info("Alle Module erfolgreich verbunden")
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise
    
def start(self, mode: str = None, auto_trade: bool = True):
try:
    self.logger.info(f"Starte Trading Bot im Modus '{mode}'...")
    self.previous_state = self.state
    self.state = BotState.RUNNING

    # Module starten
    # Datenpipeline starten (für Marktdaten)
    self.data_pipeline.start_auto_updates()
    self.module_status['data_pipeline']['status'] = "running"
    
    # Black Swan Detector starten
    self.black_swan_detector.start_monitoring()
    self.module_status['black_swan_detector']['status'] = "running"

    # Telegram-Bot starten
    self.telegram_interface.start()
    self.module_status['telegram_interface']['status'] = "running"
    
    # Live Trading starten (falls aktiviert)
    current_mode = mode or self.config.get('trading', {}).get('mode', 'paper')
    if auto_trade and current_mode != 'disabled':
        if hasattr(self.live_trading, 'is_ready') and self.live_trading.is_ready:
            self.live_trading.start_trading(mode=current_mode)
            self.module_status['live_trading']['status'] = "running"
            self.logger.info(f"Live Trading aktiviert im Modus '{current_mode}'")
        else:
            self.logger.warning("Live Trading nicht bereit, Trading wird deaktiviert")
            self.module_status['live_trading']['status'] = "disabled"
    else:
        self.logger.info("Automatisches Trading deaktiviert")
        self.module_status['live_trading']['status'] = "disabled"
            
            # Hauptüberwachungs-Thread starten
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            # Monitor-Thread starten
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Trading Bot erfolgreich gestartet")
            
            # Event für Botstart hinzufügen
            self._add_event("system", "Bot gestartet", {
                "mode": mode, 
                "auto_trade": auto_trade
            })
            
            # Bot-Start-Benachrichtigung senden
            self._send_notification(
                "Bot gestartet",
                f"Modus: {current_mode}\nTrading: {'Aktiviert' if auto_trade else 'Deaktiviert'}"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstart", {"error": str(e)})
            return False
    
    def train_models(self):
        """
        Trainiert die Modelle des Learning-Moduls.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not hasattr(self, 'learning_module'):
            self.logger.error("Learning Module nicht initialisiert")
            return False
        
        try:
            self.logger.info("Starte Modelltraining...")
            training_result = self.learning_module.train_all_models()
            
            # Event für Training hinzufügen
            self._add_event("learning", "Modelltraining durchgeführt", training_result)
            
            self.logger.info(f"Modelltraining abgeschlossen: {training_result}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Modelltraining: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._add_event("error", "Fehler beim Modelltraining", {"error": str(e)})
            return False
    
    def process_transcript(self, transcript_path: str):
        """
        Verarbeitet ein Transkript mit dem TranscriptProcessor.
        
        Args:
            transcript_path: Pfad zum Transkript
        
        Returns:
            Ergebnisdictionary der Transkriptverarbeitung
        """
        return self._process_transcript(transcript_path)
    
    def stop(self):
        """
        Stoppt den Trading Bot.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state not in [BotState.RUNNING, BotState.PAUSED]:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Stoppe Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.STOPPING
            
            # Module stoppen
            # Live Trading stoppen
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "stopped"
            
            # Black Swan Detector stoppen
            self.black_swan_detector.stop_monitoring()
            self.module_status['black_swan_detector']['status'] = "stopped"
            
            # Datenpipeline stoppen
            self.data_pipeline.stop_auto_updates()
            self.module_status['data_pipeline']['status'] = "stopped"
            
            # Hauptschleife beenden
            self.running = False
            
            # Threads beenden
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=10)
            
            # Telegram-Bot weiterlaufen lassen für Remote-Steuerung
            
            self.state = BotState.READY
            self.logger.info("Trading Bot erfolgreich gestoppt")
            
            # Event für Botstopp hinzufügen
            self._add_event("system", "Bot gestoppt", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot gestoppt", "Trading-Aktivitäten wurden beendet")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstopp", {"error": str(e)})
            return False
    
    def pause(self):
        """
        Pausiert den Trading Bot (beendet das Trading, behält aber die Überwachung bei).
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state != BotState.RUNNING:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Pausiere Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.PAUSED
            
            # Nur Trading pausieren, andere Module weiterlaufen lassen
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "paused"
            
            self.logger.info("Trading Bot erfolgreich pausiert")
            
            # Event für Botpause hinzufügen
            self._add_event("system", "Bot pausiert", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot pausiert", "Trading wurde pausiert, Überwachung bleibt aktiv")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Pausieren des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Pausieren", {"error": str(e)})
            return False
    
    def resume(self):
        """
        Setzt den pausierten Trading Bot fort.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state != BotState.PAUSED:
            self.logger.warning(f"Bot ist nicht pausiert (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Setze Trading Bot fort...")
            self.previous_state = self.state
            self.state = BotState.RUNNING
            
            # Trading wieder aktivieren
            if self.module_status['live_trading']['status'] == "paused":
                self.live_trading.start_trading()
                self.module_status['live_trading']['status'] = "running"
            
            self.logger.info("Trading Bot erfolgreich fortgesetzt")
            
            # Event für Botfortsetzung hinzufügen
            self._add_event("system", "Bot fortgesetzt", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot fortgesetzt", "Trading wurde wieder aktiviert")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Fortsetzen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Fortsetzen", {"error": str(e)})
            return False
    
    def restart(self):
        """
        Startet den Trading Bot neu.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            self.logger.info("Starte Trading Bot neu...")
            
            # Bot stoppen
            success = self.stop()
            if not success:
                self.logger.error("Fehler beim Stoppen für Neustart")
                return False
            
            # Kurze Pause
            time.sleep(3)
            
            # Bot neu starten
            return self.start()
        except Exception as e:
            self.logger.error(f"Fehler beim Neustarten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Neustart", {"error": str(e)})
            return False
    
    def _main_loop(self):
        """Hauptschleife des Trading Bots."""
        self.logger.info("Hauptschleife gestartet")
        
        while self.running:
            try:
                # Events aus der Queue verarbeiten
                self._process_events()
                
                # Auf Steuerungssignale prüfen
                if self.shutdown_requested:
                    self.logger.info("Shutdown angefordert, beende Hauptschleife")
                    break
                
                if self.restart_requested:
                    self.logger.info("Neustart angefordert, beende Hauptschleife")
                    self.restart_requested = False
                    self.thread_pool.submit(self.restart)
                    break
                
                if self.pause_requested:
                    self.logger.info("Pause angefordert")
                    self.pause_requested = False
                    self.thread_pool.submit(self.pause)
                
                # Kurze Pause, um CPU-Last zu reduzieren
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Fehler in der Hauptschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                self._add_event("error", "Fehler in Hauptschleife", {"error": str(e)})
                time.sleep(5)  # Längere Pause bei Fehlern
        
        self.logger.info("Hauptschleife beendet")
    
    def _process_events(self):
        """Verarbeitet Ereignisse aus der Event-Queue."""
        try:
            # Bis zu 10 Events pro Durchlauf verarbeiten
            for _ in range(10):
                try:
                    event = self.event_queue.get_nowait()
                    event_type = event.get('type')
                    event_data = event.get('data', {})
                    
                    if event_type == 'black_swan':
                        self._handle_black_swan_event(event_data)
                    elif event_type == 'trade':
                        self._handle_trade_event(event_data)
                    elif event_type == 'order':
                        self._handle_order_update(event_data)
                    elif event_type == 'position':
                        self._handle_position_update(event_data)
                    elif event_type == 'error':
                        self._handle_error_event(event_data)
                    elif event_type == 'command':
                        self._handle_command_event(event_data)
                    else:
                        self.logger.warning(f"Unbekannter Event-Typ: {event_type}")
                    
                    # Event als verarbeitet markieren
                    self.event_queue.task_done()
                except queue.Empty:
                    break  # Queue ist leer
        except Exception as e:
            self.logger.error(f"Fehler bei der Event-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _monitor_loop(self):
        """Überwachungsschleife für Systemzustand und Modulstatus."""
        self.logger.info("Überwachungsschleife gestartet")
        check_interval = 30  # Sekunden
        health_check_counter = 0
        
        while self.running:
            try:
                # Module auf Fehler prüfen
                for module_name, module in self.modules.items():
                    if hasattr(module, 'get_status'):
                        status = module.get_status()
                        
                        # Status aktualisieren
                        if isinstance(status, dict):
                            self.module_status[module_name]['last_status'] = status
                            
                            # Auf Fehler prüfen
                            if 'error' in status and status.get('error'):
                                self.module_status[module_name]['errors'].append({
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'error': status.get('error')
                                })
                                self.logger.warning(f"Fehler in Modul {module_name}: {status.get('error')}")
                
                # Alle 5 Durchläufe (ca. 2.5 Minuten) einen umfassenderen Health-Check durchführen
                health_check_counter += 1
                if health_check_counter >= 5:
                    health_check_counter = 0
                    self._perform_health_check()
                
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Fehler in der Überwachungsschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                time.sleep(check_interval * 2)  # Längere Pause bei Fehlern
        
        self.logger.info("Überwachungsschleife beendet")
    
    def _perform_health_check(self):
        """Führt einen umfassenden Health-Check des Systems durch."""
        self.logger.debug("Führe System-Health-Check durch...")
        
        try:
            # Prüfen, ob alle Module noch funktionieren
            for module_name, module in self.modules.items():
                # Spezifische Prüfungen je nach Modul
                if module_name == 'data_pipeline':
                    # Prüfen, ob Daten aktuell sind
                    if hasattr(self.data_pipeline, 'get_last_update_time'):
                        last_update = self.data_pipeline.get_last_update_time('crypto')
                        if last_update:
                            time_diff = (datetime.datetime.now() - last_update).total_seconds()
                            if time_diff > 300:  # Älter als 5 Minuten
                                self.logger.warning(f"Daten für 'crypto' sind veraltet ({time_diff:.0f} Sekunden)")
                
                elif module_name == 'live_trading':
                    # Prüfen, ob Verbindung zur Börse besteht
                    if self.module_status['live_trading']['status'] == "running":
                        if hasattr(self.live_trading, 'get_status'):
                            status = self.live_trading.get_status()
                            if status.get('exchange_status') != 'connected':
                                self.logger.warning(f"Live Trading nicht verbunden: {status.get('exchange_status')}")
            
            self.logger.debug("System-Health-Check abgeschlossen")
        except Exception as e:
            self.logger.error(f"Fehler beim Health-Check: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _handle_black_swan_event(self, event_data: Dict[str, Any]):
        """
        Verarbeitet ein Black Swan Ereignis.
        
        Args:
            event_data: Ereignisdaten vom Black Swan Detector
        """
        severity = event_data.get('severity', 0)
        title = event_data.get('title', 'Black Swan Event')
        message = event_data.get('message', 'Unbekanntes Marktereignis erkannt')
        details = event_data.get('details', {})
        
        self.logger.warning(f"Black Swan Event erkannt: {title} (Schweregrad: {severity:.2f})")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("black_swan", title, {
            "severity": severity,
            "message": message,
            "details": details
        })
        
        # Notfallmaßnahmen je nach Schweregrad
        if severity > 0.8:
            # Kritischer Schweregrad - Notfallmaßnahmen einleiten
            self._emergency_shutdown(message)
        elif severity > 0.5:
            # Hoher Schweregrad - Trading pausieren und Benachrichtigung senden
            if self.state == BotState.RUNNING:
                self.pause()
            
            # Benachrichtigung mit hoher Priorität senden
            self._send_notification(
                f"⚠️ KRITISCHES MARKTEREIGNIS: {title}",
                message,
                priority="high"
            )
        else:
            # Moderater Schweregrad - Nur Benachrichtigung senden
            self._send_notification(
                f"⚠️ Ungewöhnliches Marktereignis: {title}",
                message
            )
    
    def _emergency_shutdown(self, reason: str):
        """
        Führt einen Notfall-Shutdown des Systems durch.
        
        Args:
            reason: Grund für den Notfall-Shutdown
        """
        self.logger.critical(f"NOTFALL-SHUTDOWN eingeleitet: {reason}")
        
        try:
            # Status aktualisieren
            self.previous_state = self.state
            self.state = BotState.EMERGENCY
            self.emergency_mode = True
            
            # Alle Positionen schließen
            if (self.module_status['live_trading']['status'] == "running" and
                    hasattr(self.live_trading, 'close_all_positions')):
                self.logger.critical("Schließe alle Positionen...")
                try:
                    result = self.live_trading.close_all_positions()
                    self.logger.info(f"Positionen geschlossen: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Schließen aller Positionen: {str(e)}")
            
            # Alle offenen Orders stornieren
            if (self.module_status['live_trading']['status'] == "running" and
                    hasattr(self.live_trading, 'cancel_all_orders')):
                self.logger.critical("Storniere alle offenen Orders...")
                try:
                    result = self.live_trading.cancel_all_orders()
                    self.logger.info(f"Orders storniert: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Stornieren aller Orders: {str(e)}")
            
            # Trading deaktivieren
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "emergency_stopped"
            
            # DRINGENDE Benachrichtigung senden
            self._send_notification(
                "🚨 NOTFALL-SHUTDOWN AKTIVIERT 🚨",
                f"Grund: {reason}\n\nAlle Positionen wurden geschlossen und das Trading wurde deaktiviert.",
                priority="critical"
            )
            
            # Event hinzufügen
            self._add_event("emergency", "Notfall-Shutdown", {"reason": reason})
        except Exception as e:
            self.logger.error(f"Fehler beim Notfall-Shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _handle_trading_error(self, error_data: Dict[str, Any]):
        """
        Verarbeitet einen Trading-Fehler.
        
        Args:
            error_data: Fehlerdaten vom Trading-Modul
        """
        message = error_data.get('message', 'Unbekannter Trading-Fehler')
        context = error_data.get('context', '')
        consecutive_errors = error_data.get('consecutive_errors', 0)
        
        self.logger.error(f"Trading-Fehler: {message} (Kontext: {context})")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("error", "Trading-Fehler", {
            "message": message,
            "context": context,
            "consecutive_errors": consecutive_errors
        })
        
        # Bei zu vielen aufeinanderfolgenden Fehlern Trading pausieren
        if consecutive_errors >= 5:
            self.logger.warning(f"Zu viele aufeinanderfolgende Fehler ({consecutive_errors}), pausiere Trading")
            if self.state == BotState.RUNNING:
                self.pause()
            
            # Benachrichtigung senden
            self._send_notification(
                "🛑 Trading automatisch pausiert",
                f"Grund: Zu viele Fehler in Folge ({consecutive_errors})\nLetzter Fehler: {message}",
                priority="high"
            )
    
    def _handle_order_update(self, order_data: Dict[str, Any]):
        """
        Verarbeitet ein Order-Update.
        
        Args:
            order_data: Order-Daten vom Trading-Modul
        """
        order_id = order_data.get('id', 'unknown')
        symbol = order_data.get('symbol', 'unknown')
        status = order_data.get('status', 'unknown')
        
        self.logger.info(f"Order-Update: {order_id} für {symbol} - Status: {status}")
        
        # Order-Update an das Steuermodul weiterleiten
        if hasattr(self.tax_module, 'process_order'):
            self.tax_module.process_order(order_data)
        
        # Bei abgeschlossenen Orders Benachrichtigung senden
        if status == 'closed':
            side = order_data.get('side', 'unknown')
            amount = order_data.get('amount', 0)
            price = order_data.get('price', 0)
            cost = order_data.get('cost', 0)
            
            self._send_notification(
                f"Order ausgeführt: {symbol}",
                f"ID: {order_id}\nTyp: {side}\nMenge: {amount}\nPreis: {price}\nWert: {cost}",
                priority="low"
            )
        
        # Ereignis zur Historie hinzufügen
        self._add_event("order", f"Order {status}", order_data)
    
    def _handle_position_update(self, position_data: Dict[str, Any]):
        """
        Verarbeitet ein Positions-Update.
        
        Args:
            position_data: Positions-Daten vom Trading-Modul
        """
        symbol = position_data.get('symbol', 'unknown')
        action = position_data.get('action', 'unknown')
        
        self.logger.info(f"Positions-Update: {symbol} - Aktion: {action}")
        
        # Benachrichtigung bei geschlossenen Positionen senden
        if action == 'close':
            side = position_data.get('side', 'unknown')
            contracts_before = position_data.get('contracts_before', 0)
            pnl = position_data.get('pnl', 0)
            pnl_percent = position_data.get('pnl_percent', 0)
            
            message = (
                f"Richtung: {side}\n"
                f"Kontrakte: {contracts_before}\n"
                f"PnL: {pnl:.2f} ({pnl_percent:.2f}%)"
            )
            
            # Priorität basierend auf Gewinn/Verlust
            priority = "normal"
            if pnl > 0:
                title = f"Position mit Gewinn geschlossen: {symbol}"
            else:
                title = f"Position mit Verlust geschlossen: {symbol}"
                if pnl_percent < -5:
                    priority = "high"
            
            self._send_notification(title, message, priority=priority)
        
        # Bei neuen Positionen ebenfalls informieren
        elif action == 'open':
            side = position_data.get('side', 'unknown')
            contracts = position_data.get('contracts', 0)
            entry_price = position_data.get('entry_price', 0)
            leverage = position_data.get('leverage', 1)
            
            self._send_notification(
                f"Neue Position eröffnet: {symbol}",
                f"Richtung: {side}\nKontrakte: {contracts}\nEinstiegspreis: {entry_price}\nHebel: {leverage}x",
                priority="normal"
            )
        
        # Ereignis zur Historie hinzufügen
        self._add_event("position", f"Position {action}", position_data)
    
    def _handle_error_event(self, error_data: Dict[str, Any]):
        """
        Verarbeitet ein Fehler-Ereignis.
        
        Args:
            error_data: Fehlerdaten
        """
        module = error_data.get('module', 'unknown')
        message = error_data.get('message', 'Unbekannter Fehler')
        level = error_data.get('level', 'error')
        
        if level == 'critical':
            self.logger.critical(f"Kritischer Fehler in {module}: {message}")
        else:
            self.logger.error(f"Fehler in {module}: {message}")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("error", f"Fehler in {module}", error_data)
        
        # Bei kritischen Fehlern Benachrichtigung senden
        if level == 'critical':
            self._send_notification(
                f"Kritischer Fehler in {module}",
                message,
                priority="high"
            )
    
    def _handle_command_event(self, command_data: Dict[str, Any]):
        """
        Verarbeitet ein Kommando-Ereignis.
        
        Args:
            command_data: Kommandodaten
        """
        command = command_data.get('command', '')
        params = command_data.get('params', {})
        source = command_data.get('source', 'unknown')
        
        self.logger.info(f"Kommando empfangen: {command} von {source}")
        
        # Kommando ausführen
        if command == 'start':
            self.start()
        elif command == 'stop':
            self.stop()
        elif command == 'pause':
            self.pause()
        elif command == 'resume':
            self.resume()
        elif command == 'restart':
            self.restart()
        elif command == 'process_transcript':
            transcript_path = params.get('path', '')
            if transcript_path:
                self._process_transcript(transcript_path)
        else:
            self.logger.warning(f"Unbekanntes Kommando: {command}")
    
    def _handle_trade_event(self, trade_data: Dict[str, Any]):
        """
        Verarbeitet ein Trade-Ereignis.
        
        Args:
            trade_data: Trade-Daten
        """
        symbol = trade_data.get('symbol', 'unknown')
        side = trade_data.get('side', 'unknown')
        price = trade_data.get('price', 0)
        amount = trade_data.get('amount', 0)
        
        self.logger.info(f"Trade ausgeführt: {symbol} {side} {amount} @ {price}")
        
        # Trade an das Steuermodul weiterleiten
        if hasattr(self.tax_module, 'process_trade'):
            self.tax_module.process_trade(trade_data)
        
        # Ereignis zur Historie hinzufügen
        self._add_event("trade", "Trade ausgeführt", trade_data)
    
    def _send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sendet eine Benachrichtigung an alle Benachrichtigungskanäle.
        
        Args:
            title: Titel der Benachrichtigung
            message: Nachrichtentext
            priority: Priorität ('low', 'normal', 'high', 'critical')
        """
        # Nachricht formatieren
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"{message}\n\nZeit: {timestamp}"
        
        # An Telegram senden
        if hasattr(self.telegram_interface, 'send_message'):
            try:
                self.telegram_interface.send_notification(title, formatted_message, priority)
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Telegram-Benachrichtigung: {str(e)}")
        
        # In Zukunft könnten hier weitere Benachrichtigungskanäle hinzugefügt werden
        # z.B. E-Mail, Push-Benachrichtigungen, etc.
    
    def _add_event(self, event_type: str, title: str, data: Dict[str, Any]):
        """
        Fügt ein Ereignis zur Historie hinzu.
        
        Args:
            event_type: Typ des Ereignisses
            title: Titel des Ereignisses
            data: Ereignisdaten
        """
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': event_type,
            'title': title,
            'data': data
        }
        
        # Ereignis zur Historie hinzufügen
        self.events.append(event)
        
        # Historie begrenzen
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def _get_account_balance(self):
        """
        Ruft den aktuellen Kontostand ab.
        
        Returns:
            Kontostand als Dictionary oder Fehlermeldung
        """
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                    hasattr(self.live_trading, 'get_account_balance')):
                balance = self.live_trading.get_account_balance()
                return {
                    'status': 'success',
                    'balance': balance
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Live Trading ist nicht aktiv'
                }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _get_open_positions(self):
        """
        Ruft die offenen Positionen ab.
        
        Returns:
            Offene Positionen als Dictionary oder Fehlermeldung
        """
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                    hasattr(self.live_trading, 'get_open_positions')):
                positions = self.live_trading.get_open_positions()
                return {
                    'status': 'success',
                    'positions': positions
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Live Trading ist nicht aktiv'
                }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der offenen Positionen: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _get_performance_metrics(self):
        """
        Ruft die Performance-Metriken ab.
        
        Returns:
            Performance-Metriken als Dictionary
        """
        try:
            metrics = {}
            
            # Metriken vom Learning-Modul abrufen
            if hasattr(self.learning_module, 'performance_metrics'):
                metrics['learning'] = self.learning_module.performance_metrics
            
            # Handelsergebnisse abrufen
            if hasattr(self.learning_module, 'trade_history'):
                # Einfache Statistiken berechnen
                trades = self.learning_module.trade_history
                closed_trades = [t for t in trades if t.status == 'closed']
                
                if closed_trades:
                    winning_trades = [t for t in closed_trades if t.pnl_percent is not None and t.pnl_percent > 0]
                    losing_trades = [t for t in closed_trades if t.pnl_percent is not None and t.pnl_percent <= 0]
                    
                    metrics['trading'] = {
                        'total_trades': len(closed_trades),
                        'winning_trades': len(winning_trades),
                        'losing_trades': len(losing_trades),
                        'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
                        'avg_win': sum(t.pnl_percent for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                        'avg_loss': sum(t.pnl_percent for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                        'total_pnl': sum(t.pnl_percent for t in closed_trades if t.pnl_percent is not None)
                    }
            
            # Steuerliche Informationen
            if hasattr(self.tax_module, 'get_tax_summary'):
                metrics['tax'] = self.tax_module.get_tax_summary()
            
            return {
                'status': 'success',
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Metriken: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _process_transcript_command(self, params: Dict[str, Any]):
        """
        Verarbeitet ein Transkript-Verarbeitungskommando.
        
        Args:
            params: Kommandoparameter
        
        Returns:
            Ergebnis als Dictionary
        """
        transcript_path = params.get('path', '')
        if not transcript_path:
            return {
                'status': 'error',
                'message': 'Kein Transkript-Pfad angegeben'
            }
        
        return self._process_transcript(transcript_path)
    
    def _process_transcript(self, transcript_path: str):
        """
        Verarbeitet ein Transkript mit dem TranscriptProcessor.
        
        Args:
            transcript_path: Pfad zum Transkript
        
        Returns:
            Ergebnis als Dictionary
        """
        try:
            self.logger.info(f"Verarbeite Transkript: {transcript_path}")
            
            # Prüfen, ob Datei existiert
            if not os.path.exists(transcript_path):
                return {
                    'status': 'error',
                    'message': f"Transkript-Datei nicht gefunden: {transcript_path}"
                }
            
            # Transkript verarbeiten
            if hasattr(self.transcript_processor, 'process_transcript'):
                result = self.transcript_processor.process_transcript(transcript_path)
                
                # Ereignis zur Historie hinzufügen
                self._add_event("transcript", "Transkript verarbeitet", {
                    'path': transcript_path,
                    'result': result
                })
                
                # Erfolgsmeldung
                self._send_notification(
                    "Transkript verarbeitet",
                    f"Pfad: {transcript_path}\nErgebnis: {result.get('status', 'Unbekannt')}",
                    priority="normal"
                )
                
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'TranscriptProcessor unterstützt process_transcript nicht'
                }
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Status des Trading Bots zurück.
        
        Returns:
            Status als Dictionary
        """
        status = {
            'state': self.state,
            'previous_state': self.previous_state,
            'emergency_mode': self.emergency_mode,
            'running': self.running,
            'modules': self.module_status,
            'last_update': datetime.datetime.now().isoformat(),
            'events': self.events[-10:],  # Letzten 10 Ereignisse
            'version': '1.0.0',  # Bot-Version
            'uptime': self._get_uptime()
        }
        
        return status
    
    def _get_uptime(self) -> str:
        """
        Berechnet die Laufzeit des Bots.
        
        Returns:
            Laufzeit als formatierter String
        """
        # In einer vollständigen Implementierung würde hier die tatsächliche Laufzeit berechnet
        return "00:00:00"  # Dummy-Wert
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Erstellt einen umfassenden Status- und Performance-Bericht.
        
        Returns:
            Bericht als Dictionary
        """
        try:
            # Basis-Status
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'status': self.get_status(),
                'performance': self._get_performance_metrics().get('metrics', {}),
                'account': self._get_account_balance().get('balance', {}),
                'positions': self._get_open_positions().get('positions', []),
                'recent_events': self.events[-20:]  # Letzten 20 Ereignisse
            }
            
            # Learning-Modul-Status
            if hasattr(self.learning_module, 'get_current_status'):
                report['learning_status'] = self.learning_module.get_current_status()
            
            # Black Swan Detector Status
            if hasattr(self.black_swan_detector, 'get_current_status'):
                report['black_swan_status'] = self.black_swan_detector.get_current_status()
            
            # Marktdaten-Status
            if hasattr(self.data_pipeline, 'get_status'):
                report['data_status'] = self.data_pipeline.get_status()
            
            return report
        except Exception as e:
            self.logger.error(f"Fehler bei der Bericht-Generierung: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}",
                'timestamp': datetime.datetime.now().isoformat()
            }


# Beispiel für die Ausführung
if __name__ == "__main__":
    try:
        # MainController initialisieren
        controller = MainController()
        
        # Bot starten
        if controller.state == BotState.READY:
            controller.start(auto_trade=False)  # Nur im Paper-Modus
            
            # Endlosschleife, um den Bot laufen zu lassen
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nProgramm wird beendet...")
                controller.stop()
        else:
            print(f"Bot konnte nicht gestartet werden. Status: {controller.state}")
    except Exception as e:
        print(f"Kritischer Fehler: {str(e)}")
        traceback.print_exc()

import time
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import traceback
import signal
import importlib
import queue
from concurrent.futures import ThreadPoolExecutor

# Module importieren
from src.core.config_manager import ConfigManager
from src.modules.data_pipeline import DataPipeline
from src.modules.live_trading import LiveTradingConnector
from src.modules.learning_module import LearningModule
from src.modules.transcript_processor import TranscriptProcessor
from src.modules.black_swan_detector import BlackSwanDetector
from src.modules.telegram_interface import TelegramInterface
from src.modules.tax_module import TaxModule

# Logger einrichten (ohne basicConfig, da dies bereits in main.py erfolgt)
logger = logging.getLogger("MainController")

class BotState:
    """Status des Trading Bots"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class MainController:
    """
    Hauptcontroller für den Trading Bot.
    Koordiniert und verwaltet alle Module und steuert den Gesamtbetrieb.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialisiert den MainController.
        
        Args:
            config_manager: Ein ConfigManager-Objekt für die Konfigurationsverwaltung (optional)
        """
        self.logger = logging.getLogger("MainController")
        self.logger.info("Initialisiere MainController...")
        
        # Verwende den übergebenen ConfigManager
        self.config_manager = config_manager
        
        # Bot-Status
        self.state = BotState.INITIALIZING
        self.previous_state = None
        self.emergency_mode = False
        
        # Ereignisprotokollierung
        self.events = []
        self.max_events = 1000  # Maximale Anzahl der gespeicherten Ereignisse
        
        # Steuerungs-Flags
        self.running = False
        self.shutdown_requested = False
        self.restart_requested = False
        self.pause_requested = False
        
        # Threads
        self.main_thread = None
        self.monitor_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Event-Queue für asynchrone Kommunikation
        self.event_queue = queue.Queue()
        
        # Signal-Handler für Graceful Shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Module
        self.modules = {}
        self.module_status = {}
        
        # Konfiguration laden
        if self.config_manager:
            self.config = self.config_manager.get_config()
            
            # Log-Level anpassen
            log_level = self.config.get('general', {}).get('log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, log_level))
            
            # Pfade
            self.data_path = Path(self.config.get('general', {}).get('data_path', 'data'))
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            # Module initialisieren
            self._initialize_modules()
            
            # Status auf bereit setzen
            self.state = BotState.READY
            self.logger.info("MainController erfolgreich initialisiert")
        else:
            self.logger.error("Kein ConfigManager übergeben, MainController nicht vollständig initialisiert")
            self.config = {}
            self.data_path = Path('data')
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.state = BotState.ERROR
    
    def _signal_handler(self, sig, frame):
        """Behandelt Betriebssystem-Signale für sauberes Herunterfahren."""
        self.logger.info(f"Signal {sig} empfangen. Fahre Bot herunter...")
        self.shutdown_requested = True
        if self.state == BotState.RUNNING:
            self.stop()
    
    def _initialize_modules(self):
        """Initialisiert alle Module des Trading Bots."""
        try:
            # Dictionary zum Speichern der Module
            self.modules = {}
            self.module_status = {}
            
            # Datenpipeline
            self.logger.info("Initialisiere DataPipeline...")
            data_config = self.config_manager.get_config('data_pipeline')
            api_keys = self.config_manager.get_api_keys()
            self.data_pipeline = DataPipeline(api_keys)
            self.modules['data_pipeline'] = self.data_pipeline
            self.module_status['data_pipeline'] = {"status": "initialized", "errors": []}
            
            # Live Trading Connector
            self.logger.info("Initialisiere LiveTradingConnector...")
            trading_config = self.config_manager.get_config('trading')
            # API-Schlüssel für Bitget hinzufügen
            trading_config.update(self.config_manager.get_api_key('bitget'))
            self.live_trading = LiveTradingConnector(trading_config)
            self.modules['live_trading'] = self.live_trading
            self.module_status['live_trading'] = {"status": "initialized", "errors": []}
            
            # Learning Module
            self.logger.info("Initialisiere LearningModule...")
            learning_config = self.config_manager.get_config('learning_module')
            self.learning_module = LearningModule(learning_config)
            self.modules['learning_module'] = self.learning_module
            self.module_status['learning_module'] = {"status": "initialized", "errors": []}
            
            # Black Swan Detector
            self.logger.info("Initialisiere BlackSwanDetector...")
            blackswan_config = self.config_manager.get_config('black_swan_detector')
            self.black_swan_detector = BlackSwanDetector(blackswan_config)
            self.modules['black_swan_detector'] = self.black_swan_detector
            self.module_status['black_swan_detector'] = {"status": "initialized", "errors": []}
            
            # Telegram Interface
            self.logger.info("Initialisiere TelegramInterface...")
            telegram_config = self.config_manager.get_config('telegram')
            # Bot-Token und erlaubte Benutzer hinzufügen
            telegram_config.update(self.config_manager.get_api_key('telegram'))
            self.telegram_interface = TelegramInterface(telegram_config, self)
            self.modules['telegram_interface'] = self.telegram_interface
            self.module_status['telegram_interface'] = {"status": "initialized", "errors": []}
            
            # Transcript Processor
            self.logger.info("Initialisiere TranscriptProcessor...")
            transcript_config = self.config_manager.get_config('transcript_processor') or {}
            self.transcript_processor = TranscriptProcessor(transcript_config)
            self.modules['transcript_processor'] = self.transcript_processor
            self.module_status['transcript_processor'] = {"status": "initialized", "errors": []}
            
            # Tax Module
            self.logger.info("Initialisiere TaxModule...")
            tax_config = self.config_manager.get_config('tax_module')
            self.tax_module = TaxModule(tax_config)
            self.modules['tax_module'] = self.tax_module
            self.module_status['tax_module'] = {"status": "initialized", "errors": []}
            
            # Module miteinander verbinden
            self._connect_modules()
            
            self.logger.info("Alle Module erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise
    
    def _connect_modules(self):
        """Verbindet die Module miteinander für Kommunikation und Datenaustausch."""
        try:
            # Black Swan Detector mit Live Trading verbinden
            self.black_swan_detector.register_notification_callback(self._handle_black_swan_event)
            
            # Telegram Interface Callbacks registrieren
            telegram_commands = {
                'start': self.start,
                'stop': self.stop,
                'status': self.get_status,
                'balance': self._get_account_balance,
                'positions': self._get_open_positions,
                'performance': self._get_performance_metrics,
                'process_transcript': self._process_transcript_command
            }
            self.telegram_interface.register_commands(telegram_commands)
            
            # Live Trading Error-Callbacks registrieren
            self.live_trading.register_error_callback(self._handle_trading_error)
            self.live_trading.register_order_update_callback(self._handle_order_update)
            self.live_trading.register_position_update_callback(self._handle_position_update)
            
            # Tax Module mit Live Trading verbinden
            self.live_trading.register_order_update_callback(self.tax_module.process_trade)
            
            self.logger.info("Alle Module erfolgreich verbunden")
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise
    
    def start(self, mode: str = None, auto_trade: bool = True):
        """
        Startet den Trading Bot.
        
        Args:
            mode: Trading-Modus ('live', 'paper', 'backtest', 'learn')
            auto_trade: Ob automatisches Trading aktiviert werden soll
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state == BotState.RUNNING:
            self.logger.warning("Bot läuft bereits")
            return False
        
        if self.state == BotState.ERROR:
            self.logger.error("Bot kann aufgrund von Fehlern nicht gestartet werden")
            return False
        
        try:
            self.logger.info(f"Starte Trading Bot im Modus '{mode}'...")
            self.previous_state = self.state
            self.state = BotState.RUNNING
            
            # Module starten
            # Datenpipeline starten (für Marktdaten)
            self.data_pipeline.start_auto_updates()
            self.module_status['data_pipeline']['status'] = "running"
            
            # Black Swan Detector starten
            self.black_swan_detector.start_monitoring()
            self.module_status['black_swan_detector']['status'] = "running"
            
            # Telegram-Bot starten
            self.telegram_interface.start()
            self.module_status['telegram_interface']['status'] = "running"
            
            # Live Trading starten (falls aktiviert)
            current_mode = mode or self.config.get('trading', {}).get('mode', 'paper')
            
            if auto_trade and current_mode != 'disabled':
                if hasattr(self.live_trading, 'is_ready') and self.live_trading.is_ready:
                    self.live_trading.start_trading(mode=current_mode)
                    self.module_status['live_trading']['status'] = "running"
                    self.logger.info(f"Live Trading aktiviert im Modus '{current_mode}'")
                else:
                    self.logger.warning("Live Trading nicht bereit, Trading wird deaktiviert")
                    self.module_status['live_trading']['status'] = "disabled"
            else:
                self.logger.info("Automatisches Trading deaktiviert")
                self.module_status['live_trading']['status'] = "disabled"
            
            # Hauptüberwachungs-Thread starten
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            # Monitor-Thread starten
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Trading Bot erfolgreich gestartet")
            
            # Event für Botstart hinzufügen
            self._add_event("system", "Bot gestartet", {
                "mode": mode, 
                "auto_trade": auto_trade
            })
            
            # Bot-Start-Benachrichtigung senden
            self._send_notification(
                "Bot gestartet",
                f"Modus: {current_mode}\nTrading: {'Aktiviert' if auto_trade else 'Deaktiviert'}"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstart", {"error": str(e)})
            return False
    
    def train_models(self):
        """
        Trainiert die Modelle des Learning-Moduls.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not hasattr(self, 'learning_module'):
            self.logger.error("Learning Module nicht initialisiert")
            return False
        
        try:
            self.logger.info("Starte Modelltraining...")
            training_result = self.learning_module.train_all_models()
            
            # Event für Training hinzufügen
            self._add_event("learning", "Modelltraining durchgeführt", training_result)
            
            self.logger.info(f"Modelltraining abgeschlossen: {training_result}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Modelltraining: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._add_event("error", "Fehler beim Modelltraining", {"error": str(e)})
            return False
    
    def process_transcript(self, transcript_path: str):
        """
        Verarbeitet ein Transkript mit dem TranscriptProcessor.
        
        Args:
            transcript_path: Pfad zum Transkript
        
        Returns:
            Ergebnisdictionary der Transkriptverarbeitung
        """
        return self._process_transcript(transcript_path)
    
    def stop(self):
        """
        Stoppt den Trading Bot.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state not in [BotState.RUNNING, BotState.PAUSED]:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Stoppe Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.STOPPING
            
            # Module stoppen
            # Live Trading stoppen
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "stopped"
            
            # Black Swan Detector stoppen
            self.black_swan_detector.stop_monitoring()
            self.module_status['black_swan_detector']['status'] = "stopped"
            
            # Datenpipeline stoppen
            self.data_pipeline.stop_auto_updates()
            self.module_status['data_pipeline']['status'] = "stopped"
            
            # Hauptschleife beenden
            self.running = False
            
            # Threads beenden
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=10)
            
            # Telegram-Bot weiterlaufen lassen für Remote-Steuerung
            
            self.state = BotState.READY
            self.logger.info("Trading Bot erfolgreich gestoppt")
            
            # Event für Botstopp hinzufügen
            self._add_event("system", "Bot gestoppt", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot gestoppt", "Trading-Aktivitäten wurden beendet")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstopp", {"error": str(e)})
            return False
    
    def pause(self):
        """
        Pausiert den Trading Bot (beendet das Trading, behält aber die Überwachung bei).
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state != BotState.RUNNING:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Pausiere Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.PAUSED
            
            # Nur Trading pausieren, andere Module weiterlaufen lassen
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "paused"
            
            self.logger.info("Trading Bot erfolgreich pausiert")
            
            # Event für Botpause hinzufügen
            self._add_event("system", "Bot pausiert", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot pausiert", "Trading wurde pausiert, Überwachung bleibt aktiv")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Pausieren des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Pausieren", {"error": str(e)})
            return False
    
    def resume(self):
        """
        Setzt den pausierten Trading Bot fort.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.state != BotState.PAUSED:
            self.logger.warning(f"Bot ist nicht pausiert (Status: {self.state})")
            return False
        
        try:
            self.logger.info("Setze Trading Bot fort...")
            self.previous_state = self.state
            self.state = BotState.RUNNING
            
            # Trading wieder aktivieren
            if self.module_status['live_trading']['status'] == "paused":
                self.live_trading.start_trading()
                self.module_status['live_trading']['status'] = "running"
            
            self.logger.info("Trading Bot erfolgreich fortgesetzt")
            
            # Event für Botfortsetzung hinzufügen
            self._add_event("system", "Bot fortgesetzt", {})
            
            # Benachrichtigung senden
            self._send_notification("Bot fortgesetzt", "Trading wurde wieder aktiviert")
            
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Fortsetzen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Fortsetzen", {"error": str(e)})
            return False
    
    def restart(self):
        """
        Startet den Trading Bot neu.
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            self.logger.info("Starte Trading Bot neu...")
            
            # Bot stoppen
            success = self.stop()
            if not success:
                self.logger.error("Fehler beim Stoppen für Neustart")
                return False
            
            # Kurze Pause
            time.sleep(3)
            
            # Bot neu starten
            return self.start()
        except Exception as e:
            self.logger.error(f"Fehler beim Neustarten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Neustart", {"error": str(e)})
            return False
    
    def _main_loop(self):
        """Hauptschleife des Trading Bots."""
        self.logger.info("Hauptschleife gestartet")
        
        while self.running:
            try:
                # Events aus der Queue verarbeiten
                self._process_events()
                
                # Auf Steuerungssignale prüfen
                if self.shutdown_requested:
                    self.logger.info("Shutdown angefordert, beende Hauptschleife")
                    break
                
                if self.restart_requested:
                    self.logger.info("Neustart angefordert, beende Hauptschleife")
                    self.restart_requested = False
                    self.thread_pool.submit(self.restart)
                    break
                
                if self.pause_requested:
                    self.logger.info("Pause angefordert")
                    self.pause_requested = False
                    self.thread_pool.submit(self.pause)
                
                # Kurze Pause, um CPU-Last zu reduzieren
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Fehler in der Hauptschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                self._add_event("error", "Fehler in Hauptschleife", {"error": str(e)})
                time.sleep(5)  # Längere Pause bei Fehlern
        
        self.logger.info("Hauptschleife beendet")
    
    def _process_events(self):
        """Verarbeitet Ereignisse aus der Event-Queue."""
        try:
            # Bis zu 10 Events pro Durchlauf verarbeiten
            for _ in range(10):
                try:
                    event = self.event_queue.get_nowait()
                    event_type = event.get('type')
                    event_data = event.get('data', {})
                    
                    if event_type == 'black_swan':
                        self._handle_black_swan_event(event_data)
                    elif event_type == 'trade':
                        self._handle_trade_event(event_data)
                    elif event_type == 'order':
                        self._handle_order_update(event_data)
                    elif event_type == 'position':
                        self._handle_position_update(event_data)
                    elif event_type == 'error':
                        self._handle_error_event(event_data)
                    elif event_type == 'command':
                        self._handle_command_event(event_data)
                    else:
                        self.logger.warning(f"Unbekannter Event-Typ: {event_type}")
                    
                    # Event als verarbeitet markieren
                    self.event_queue.task_done()
                except queue.Empty:
                    break  # Queue ist leer
        except Exception as e:
            self.logger.error(f"Fehler bei der Event-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _monitor_loop(self):
        """Überwachungsschleife für Systemzustand und Modulstatus."""
        self.logger.info("Überwachungsschleife gestartet")
        check_interval = 30  # Sekunden
        health_check_counter = 0
        
        while self.running:
            try:
                # Module auf Fehler prüfen
                for module_name, module in self.modules.items():
                    if hasattr(module, 'get_status'):
                        status = module.get_status()
                        
                        # Status aktualisieren
                        if isinstance(status, dict):
                            self.module_status[module_name]['last_status'] = status
                            
                            # Auf Fehler prüfen
                            if 'error' in status and status.get('error'):
                                self.module_status[module_name]['errors'].append({
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'error': status.get('error')
                                })
                                self.logger.warning(f"Fehler in Modul {module_name}: {status.get('error')}")
                
                # Alle 5 Durchläufe (ca. 2.5 Minuten) einen umfassenderen Health-Check durchführen
                health_check_counter += 1
                if health_check_counter >= 5:
                    health_check_counter = 0
                    self._perform_health_check()
                
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Fehler in der Überwachungsschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                time.sleep(check_interval * 2)  # Längere Pause bei Fehlern
        
        self.logger.info("Überwachungsschleife beendet")
    
    def _perform_health_check(self):
        """Führt einen umfassenden Health-Check des Systems durch."""
        self.logger.debug("Führe System-Health-Check durch...")
        
        try:
            # Prüfen, ob alle Module noch funktionieren
            for module_name, module in self.modules.items():
                # Spezifische Prüfungen je nach Modul
                if module_name == 'data_pipeline':
                    # Prüfen, ob Daten aktuell sind
                    if hasattr(self.data_pipeline, 'get_last_update_time'):
                        last_update = self.data_pipeline.get_last_update_time('crypto')
                        if last_update:
                            time_diff = (datetime.datetime.now() - last_update).total_seconds()
                            if time_diff > 300:  # Älter als 5 Minuten
                                self.logger.warning(f"Daten für 'crypto' sind veraltet ({time_diff:.0f} Sekunden)")
                
                elif module_name == 'live_trading':
                    # Prüfen, ob Verbindung zur Börse besteht
                    if self.module_status['live_trading']['status'] == "running":
                        if hasattr(self.live_trading, 'get_status'):
                            status = self.live_trading.get_status()
                            if status.get('exchange_status') != 'connected':
                                self.logger.warning(f"Live Trading nicht verbunden: {status.get('exchange_status')}")
            
            self.logger.debug("System-Health-Check abgeschlossen")
        except Exception as e:
            self.logger.error(f"Fehler beim Health-Check: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _handle_black_swan_event(self, event_data: Dict[str, Any]):
        """
        Verarbeitet ein Black Swan Ereignis.
        
        Args:
            event_data: Ereignisdaten vom Black Swan Detector
        """
        severity = event_data.get('severity', 0)
        title = event_data.get('title', 'Black Swan Event')
        message = event_data.get('message', 'Unbekanntes Marktereignis erkannt')
        details = event_data.get('details', {})
        
        self.logger.warning(f"Black Swan Event erkannt: {title} (Schweregrad: {severity:.2f})")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("black_swan", title, {
            "severity": severity,
            "message": message,
            "details": details
        })
        
        # Notfallmaßnahmen je nach Schweregrad
        if severity > 0.8:
            # Kritischer Schweregrad - Notfallmaßnahmen einleiten
            self._emergency_shutdown(message)
        elif severity > 0.5:
            # Hoher Schweregrad - Trading pausieren und Benachrichtigung senden
            if self.state == BotState.RUNNING:
                self.pause()
            
            # Benachrichtigung mit hoher Priorität senden
            self._send_notification(
                f"⚠️ KRITISCHES MARKTEREIGNIS: {title}",
                message,
                priority="high"
            )
        else:
            # Moderater Schweregrad - Nur Benachrichtigung senden
            self._send_notification(
                f"⚠️ Ungewöhnliches Marktereignis: {title}",
                message
            )
    
    def _emergency_shutdown(self, reason: str):
        """
        Führt einen Notfall-Shutdown des Systems durch.
        
        Args:
            reason: Grund für den Notfall-Shutdown
        """
        self.logger.critical(f"NOTFALL-SHUTDOWN eingeleitet: {reason}")
        
        try:
            # Status aktualisieren
            self.previous_state = self.state
            self.state = BotState.EMERGENCY
            self.emergency_mode = True
            
            # Alle Positionen schließen
            if (self.module_status['live_trading']['status'] == "running" and
                    hasattr(self.live_trading, 'close_all_positions')):
                self.logger.critical("Schließe alle Positionen...")
                try:
                    result = self.live_trading.close_all_positions()
                    self.logger.info(f"Positionen geschlossen: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Schließen aller Positionen: {str(e)}")
            
            # Alle offenen Orders stornieren
            if (self.module_status['live_trading']['status'] == "running" and
                    hasattr(self.live_trading, 'cancel_all_orders')):
                self.logger.critical("Storniere alle offenen Orders...")
                try:
                    result = self.live_trading.cancel_all_orders()
                    self.logger.info(f"Orders storniert: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Stornieren aller Orders: {str(e)}")
            
            # Trading deaktivieren
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "emergency_stopped"
            
            # DRINGENDE Benachrichtigung senden
            self._send_notification(
                "🚨 NOTFALL-SHUTDOWN AKTIVIERT 🚨",
                f"Grund: {reason}\n\nAlle Positionen wurden geschlossen und das Trading wurde deaktiviert.",
                priority="critical"
            )
            
            # Event hinzufügen
            self._add_event("emergency", "Notfall-Shutdown", {"reason": reason})
        except Exception as e:
            self.logger.error(f"Fehler beim Notfall-Shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def _handle_trading_error(self, error_data: Dict[str, Any]):
        """
        Verarbeitet einen Trading-Fehler.
        
        Args:
            error_data: Fehlerdaten vom Trading-Modul
        """
        message = error_data.get('message', 'Unbekannter Trading-Fehler')
        context = error_data.get('context', '')
        consecutive_errors = error_data.get('consecutive_errors', 0)
        
        self.logger.error(f"Trading-Fehler: {message} (Kontext: {context})")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("error", "Trading-Fehler", {
            "message": message,
            "context": context,
            "consecutive_errors": consecutive_errors
        })
        
        # Bei zu vielen aufeinanderfolgenden Fehlern Trading pausieren
        if consecutive_errors >= 5:
            self.logger.warning(f"Zu viele aufeinanderfolgende Fehler ({consecutive_errors}), pausiere Trading")
            if self.state == BotState.RUNNING:
                self.pause()
            
            # Benachrichtigung senden
            self._send_notification(
                "🛑 Trading automatisch pausiert",
                f"Grund: Zu viele Fehler in Folge ({consecutive_errors})\nLetzter Fehler: {message}",
                priority="high"
            )
    
    def _handle_order_update(self, order_data: Dict[str, Any]):
        """
        Verarbeitet ein Order-Update.
        
        Args:
            order_data: Order-Daten vom Trading-Modul
        """
        order_id = order_data.get('id', 'unknown')
        symbol = order_data.get('symbol', 'unknown')
        status = order_data.get('status', 'unknown')
        
        self.logger.info(f"Order-Update: {order_id} für {symbol} - Status: {status}")
        
        # Order-Update an das Steuermodul weiterleiten
        if hasattr(self.tax_module, 'process_order'):
            self.tax_module.process_order(order_data)
        
        # Bei abgeschlossenen Orders Benachrichtigung senden
        if status == 'closed':
            side = order_data.get('side', 'unknown')
            amount = order_data.get('amount', 0)
            price = order_data.get('price', 0)
            cost = order_data.get('cost', 0)
            
            self._send_notification(
                f"Order ausgeführt: {symbol}",
                f"ID: {order_id}\nTyp: {side}\nMenge: {amount}\nPreis: {price}\nWert: {cost}",
                priority="low"
            )
        
        # Ereignis zur Historie hinzufügen
        self._add_event("order", f"Order {status}", order_data)
    
    def _handle_position_update(self, position_data: Dict[str, Any]):
        """
        Verarbeitet ein Positions-Update.
        
        Args:
            position_data: Positions-Daten vom Trading-Modul
        """
        symbol = position_data.get('symbol', 'unknown')
        action = position_data.get('action', 'unknown')
        
        self.logger.info(f"Positions-Update: {symbol} - Aktion: {action}")
        
        # Benachrichtigung bei geschlossenen Positionen senden
        if action == 'close':
            side = position_data.get('side', 'unknown')
            contracts_before = position_data.get('contracts_before', 0)
            pnl = position_data.get('pnl', 0)
            pnl_percent = position_data.get('pnl_percent', 0)
            
            message = (
                f"Richtung: {side}\n"
                f"Kontrakte: {contracts_before}\n"
                f"PnL: {pnl:.2f} ({pnl_percent:.2f}%)"
            )
            
            # Priorität basierend auf Gewinn/Verlust
            priority = "normal"
            if pnl > 0:
                title = f"Position mit Gewinn geschlossen: {symbol}"
            else:
                title = f"Position mit Verlust geschlossen: {symbol}"
                if pnl_percent < -5:
                    priority = "high"
            
            self._send_notification(title, message, priority=priority)
        
        # Bei neuen Positionen ebenfalls informieren
        elif action == 'open':
            side = position_data.get('side', 'unknown')
            contracts = position_data.get('contracts', 0)
            entry_price = position_data.get('entry_price', 0)
            leverage = position_data.get('leverage', 1)
            
            self._send_notification(
                f"Neue Position eröffnet: {symbol}",
                f"Richtung: {side}\nKontrakte: {contracts}\nEinstiegspreis: {entry_price}\nHebel: {leverage}x",
                priority="normal"
            )
        
        # Ereignis zur Historie hinzufügen
        self._add_event("position", f"Position {action}", position_data)
    
    def _handle_error_event(self, error_data: Dict[str, Any]):
        """
        Verarbeitet ein Fehler-Ereignis.
        
        Args:
            error_data: Fehlerdaten
        """
        module = error_data.get('module', 'unknown')
        message = error_data.get('message', 'Unbekannter Fehler')
        level = error_data.get('level', 'error')
        
        if level == 'critical':
            self.logger.critical(f"Kritischer Fehler in {module}: {message}")
        else:
            self.logger.error(f"Fehler in {module}: {message}")
        
        # Ereignis zur Historie hinzufügen
        self._add_event("error", f"Fehler in {module}", error_data)
        
        # Bei kritischen Fehlern Benachrichtigung senden
        if level == 'critical':
            self._send_notification(
                f"Kritischer Fehler in {module}",
                message,
                priority="high"
            )
    
    def _handle_command_event(self, command_data: Dict[str, Any]):
        """
        Verarbeitet ein Kommando-Ereignis.
        
        Args:
            command_data: Kommandodaten
        """
        command = command_data.get('command', '')
        params = command_data.get('params', {})
        source = command_data.get('source', 'unknown')
        
        self.logger.info(f"Kommando empfangen: {command} von {source}")
        
        # Kommando ausführen
        if command == 'start':
            self.start()
        elif command == 'stop':
            self.stop()
        elif command == 'pause':
            self.pause()
        elif command == 'resume':
            self.resume()
        elif command == 'restart':
            self.restart()
        elif command == 'process_transcript':
            transcript_path = params.get('path', '')
            if transcript_path:
                self._process_transcript(transcript_path)
        else:
            self.logger.warning(f"Unbekanntes Kommando: {command}")
    
    def _handle_trade_event(self, trade_data: Dict[str, Any]):
        """
        Verarbeitet ein Trade-Ereignis.
        
        Args:
            trade_data: Trade-Daten
        """
        symbol = trade_data.get('symbol', 'unknown')
        side = trade_data.get('side', 'unknown')
        price = trade_data.get('price', 0)
        amount = trade_data.get('amount', 0)
        
        self.logger.info(f"Trade ausgeführt: {symbol} {side} {amount} @ {price}")
        
        # Trade an das Steuermodul weiterleiten
        if hasattr(self.tax_module, 'process_trade'):
            self.tax_module.process_trade(trade_data)
        
        # Ereignis zur Historie hinzufügen
        self._add_event("trade", "Trade ausgeführt", trade_data)
    
    def _send_notification(self, title: str, message: str, priority: str = "normal"):
        """
        Sendet eine Benachrichtigung an alle Benachrichtigungskanäle.
        
        Args:
            title: Titel der Benachrichtigung
            message: Nachrichtentext
            priority: Priorität ('low', 'normal', 'high', 'critical')
        """
        # Nachricht formatieren
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"{message}\n\nZeit: {timestamp}"
        
        # An Telegram senden
        if hasattr(self.telegram_interface, 'send_message'):
            try:
                self.telegram_interface.send_notification(title, formatted_message, priority)
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Telegram-Benachrichtigung: {str(e)}")
        
        # In Zukunft könnten hier weitere Benachrichtigungskanäle hinzugefügt werden
        # z.B. E-Mail, Push-Benachrichtigungen, etc.
    
    def _add_event(self, event_type: str, title: str, data: Dict[str, Any]):
        """
        Fügt ein Ereignis zur Historie hinzu.
        
        Args:
            event_type: Typ des Ereignisses
            title: Titel des Ereignisses
            data: Ereignisdaten
        """
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': event_type,
            'title': title,
            'data': data
        }
        
        # Ereignis zur Historie hinzufügen
        self.events.append(event)
        
        # Historie begrenzen
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def _get_account_balance(self):
        """
        Ruft den aktuellen Kontostand ab.
        
        Returns:
            Kontostand als Dictionary oder Fehlermeldung
        """
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                hasattr(self.live_trading, 'get_account_balance')):
                balance = self.live_trading.get_account_balance()
                return {
                    'status': 'success',
                    'balance': balance
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Live Trading ist nicht aktiv'
                }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _get_open_positions(self):
        """
        Ruft die offenen Positionen ab.
        
        Returns:
            Offene Positionen als Dictionary oder Fehlermeldung
        """
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                hasattr(self.live_trading, 'get_open_positions')):
                positions = self.live_trading.get_open_positions()
                return {
                    'status': 'success',
                    'positions': positions
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Live Trading ist nicht aktiv'
                }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der offenen Positionen: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _get_performance_metrics(self):
        """
        Ruft die Performance-Metriken ab.
        
        Returns:
            Performance-Metriken als Dictionary
        """
        try:
            metrics = {}
            
            # Metriken vom Learning-Modul abrufen
            if hasattr(self.learning_module, 'performance_metrics'):
                metrics['learning'] = self.learning_module.performance_metrics
            
            # Handelsergebnisse abrufen
            if hasattr(self.learning_module, 'trade_history'):
                # Einfache Statistiken berechnen
                trades = self.learning_module.trade_history
                closed_trades = [t for t in trades if t.status == 'closed']
                
                if closed_trades:
                    winning_trades = [t for t in closed_trades if t.pnl_percent is not None and t.pnl_percent > 0]
                    losing_trades = [t for t in closed_trades if t.pnl_percent is not None and t.pnl_percent <= 0]
                    
                    metrics['trading'] = {
                        'total_trades': len(closed_trades),
                        'winning_trades': len(winning_trades),
                        'losing_trades': len(losing_trades),
                        'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
                        'avg_win': sum(t.pnl_percent for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                        'avg_loss': sum(t.pnl_percent for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                        'total_pnl': sum(t.pnl_percent for t in closed_trades if t.pnl_percent is not None)
                    }
            
            # Steuerliche Informationen
            if hasattr(self.tax_module, 'get_tax_summary'):
                metrics['tax'] = self.tax_module.get_tax_summary()
            
            return {
                'status': 'success',
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Metriken: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def _process_transcript_command(self, params: Dict[str, Any]):
        """
        Verarbeitet ein Transkript-Verarbeitungskommando.
        
        Args:
            params: Kommandoparameter
        
        Returns:
            Ergebnis als Dictionary
        """
        transcript_path = params.get('path', '')
        if not transcript_path:
            return {
                'status': 'error',
                'message': 'Kein Transkript-Pfad angegeben'
            }
        
        return self._process_transcript(transcript_path)
    
    def _process_transcript(self, transcript_path: str):
        """
        Verarbeitet ein Transkript mit dem TranscriptProcessor.
        
        Args:
            transcript_path: Pfad zum Transkript
        
        Returns:
            Ergebnis als Dictionary
        """
        try:
            self.logger.info(f"Verarbeite Transkript: {transcript_path}")
            
            # Prüfen, ob Datei existiert
            if not os.path.exists(transcript_path):
                return {
                    'status': 'error',
                    'message': f"Transkript-Datei nicht gefunden: {transcript_path}"
                }
            
            # Transkript verarbeiten
            if hasattr(self.transcript_processor, 'process_transcript'):
                result = self.transcript_processor.process_transcript(transcript_path)
                
                # Ereignis zur Historie hinzufügen
                self._add_event("transcript", "Transkript verarbeitet", {
                    'path': transcript_path,
                    'result': result
                })
                
                # Erfolgsmeldung
                self._send_notification(
                    "Transkript verarbeitet",
                    f"Pfad: {transcript_path}\nErgebnis: {result.get('status', 'Unbekannt')}",
                    priority="normal"
                )
                
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'TranscriptProcessor unterstützt process_transcript nicht'
                }
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Status des Trading Bots zurück.
        
        Returns:
            Status als Dictionary
        """
        status = {
            'state': self.state,
            'previous_state': self.previous_state,
            'emergency_mode': self.emergency_mode,
            'running': self.running,
            'modules': self.module_status,
            'last_update': datetime.datetime.now().isoformat(),
            'events': self.events[-10:],  # Letzten 10 Ereignisse
            'version': '1.0.0',  # Bot-Version
            'uptime': self._get_uptime()
        }
        
        return status
    
    def _get_uptime(self) -> str:
        """
        Berechnet die Laufzeit des Bots.
        
        Returns:
            Laufzeit als formatierter String
        """
        # In einer vollständigen Implementierung würde hier die tatsächliche Laufzeit berechnet
        return "00:00:00"  # Dummy-Wert
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Erstellt einen umfassenden Status- und Performance-Bericht.
        
        Returns:
            Bericht als Dictionary
        """
        try:
            # Basis-Status
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'status': self.get_status(),
                'performance': self._get_performance_metrics().get('metrics', {}),
                'account': self._get_account_balance().get('balance', {}),
                'positions': self._get_open_positions().get('positions', []),
                'recent_events': self.events[-20:]  # Letzten 20 Ereignisse
            }
            
            # Learning-Modul-Status
            if hasattr(self.learning_module, 'get_current_status'):
                report['learning_status'] = self.learning_module.get_current_status()
            
            # Black Swan Detector Status
            if hasattr(self.black_swan_detector, 'get_current_status'):
                report['black_swan_status'] = self.black_swan_detector.get_current_status()
            
            # Marktdaten-Status
            if hasattr(self.data_pipeline, 'get_status'):
                report['data_status'] = self.data_pipeline.get_status()
            
            return report
        except Exception as e:
            self.logger.error(f"Fehler bei der Bericht-Generierung: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}",
                'timestamp': datetime.datetime.now().isoformat()
            }


# Beispiel für die Ausführung
if __name__ == "__main__":
    try:
        # MainController initialisieren
        controller = MainController()
        
        # Bot starten
        if controller.state == BotState.READY:
            controller.start(auto_trade=False)  # Nur im Paper-Modus
            
            # Endlosschleife, um den Bot laufen zu lassen
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nProgramm wird beendet...")
                controller.stop()
        else:
            print(f"Bot konnte nicht gestartet werden. Status: {controller.state}")
    except Exception as e:
        print(f"Kritischer Fehler: {str(e)}")
        traceback.print_exc()

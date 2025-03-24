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

from src.core.config_manager import ConfigManager
from src.modules.data_pipeline import DataPipeline
from src.modules.live_trading import LiveTradingConnector
from src.modules.learning_module import LearningModule
from src.modules.transcript_processor import TranscriptProcessor
from src.modules.black_swan_detector import BlackSwanDetector
from src.modules.telegram_interface import TelegramInterface
from src.modules.tax_module import TaxModule

logger = logging.getLogger("MainController")

class BotState:
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class MainController:
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger("MainController")
        self.logger.info("Initialisiere MainController...")
        self.config_manager = config_manager
        self.state = BotState.INITIALIZING
        self.previous_state = None
        self.emergency_mode = False
        self.events = []
        self.max_events = 1000
        self.running = False
        self.shutdown_requested = False
        self.restart_requested = False
        self.pause_requested = False
        self.main_thread = None
        self.monitor_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.event_queue = queue.Queue()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.modules = {}
        self.module_status = {}
        if self.config_manager:
            self.config = self.config_manager.get_config()
            log_level = self.config.get('general', {}).get('log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, log_level))
            self.data_path = Path(self.config.get('general', {}).get('data_path', 'data'))
            self.data_path.mkdir(parents=True, exist_ok=True)
            self._initialize_modules()
            self.state = BotState.READY
            self.logger.info("MainController erfolgreich initialisiert")
        else:
            self.logger.error("Kein ConfigManager übergeben, MainController nicht vollständig initialisiert")
            self.config = {}
            self.data_path = Path('data')
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.state = BotState.ERROR

    def _signal_handler(self, sig, frame):
        self.logger.info(f"Signal {sig} empfangen. Fahre Bot herunter...")
        self.shutdown_requested = True
        if self.state == BotState.RUNNING:
            self.stop()

    def _initialize_modules(self):
        try:
            self.modules = {}
            self.module_status = {}
            self.logger.info("Initialisiere DataPipeline...")
            data_config = self.config_manager.get_config('data_pipeline')
            api_keys = self.config_manager.get_api_keys()
            self.data_pipeline = DataPipeline(api_keys)
            self.modules['data_pipeline'] = self.data_pipeline
            self.module_status['data_pipeline'] = {"status": "initialized", "errors": []}
            
            self.logger.info("Initialisiere BlackSwanDetector...")
            blackswan_config = self.config_manager.get_config('black_swan_detector')
            self.black_swan_detector = BlackSwanDetector(blackswan_config)
            self.modules['black_swan_detector'] = self.black_swan_detector
            self.module_status['black_swan_detector'] = {"status": "initialized", "errors": []}
            self.black_swan_detector.set_data_pipeline(self.data_pipeline)
            
            self.logger.info("Initialisiere LiveTradingConnector...")
            trading_config = self.config_manager.get_config('trading')
            trading_config.update(self.config_manager.get_api_key('bitget'))
            self.live_trading = LiveTradingConnector(trading_config)
            self.modules['live_trading'] = self.live_trading
            self.module_status['live_trading'] = {"status": "initialized", "errors": []}
            
            self.logger.info("Initialisiere LearningModule...")
            learning_config = self.config_manager.get_config('learning_module')
            self.learning_module = LearningModule(learning_config)
            self.modules['learning_module'] = self.learning_module
            self.module_status['learning_module'] = {"status": "initialized", "errors": []}
            
            self.logger.info("Initialisiere TelegramInterface...")
            telegram_config = self.config_manager.get_config('telegram')
            telegram_config.update(self.config_manager.get_api_key('telegram'))
            self.telegram_interface = TelegramInterface(telegram_config, self)
            self.modules['telegram_interface'] = self.telegram_interface
            self.module_status['telegram_interface'] = {"status": "initialized", "errors": []}
            
            self.logger.info("Initialisiere TranscriptProcessor...")
            transcript_config = self.config_manager.get_config('transcript_processor') or {}
            self.transcript_processor = TranscriptProcessor(transcript_config)
            self.modules['transcript_processor'] = self.transcript_processor
            self.module_status['transcript_processor'] = {"status": "initialized", "errors": []}
            
            self.logger.info("Initialisiere TaxModule...")
            tax_config = self.config_manager.get_config('tax_module')
            self.tax_module = TaxModule(tax_config)
            self.modules['tax_module'] = self.tax_module
            self.module_status['tax_module'] = {"status": "initialized", "errors": []}
            
            self._connect_modules()
            self.logger.info("Alle Module erfolgreich initialisiert")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise

    def _connect_modules(self):
        try:
            self.black_swan_detector.register_notification_callback(self._handle_black_swan_event)
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
            self.live_trading.register_error_callback(self._handle_trading_error)
            self.live_trading.register_order_update_callback(self._handle_order_update)
            self.live_trading.register_position_update_callback(self._handle_position_update)
            self.live_trading.register_order_update_callback(self.tax_module.process_trade)
            self.logger.info("Alle Module erfolgreich verbunden")
        except Exception as e:
            self.logger.error(f"Fehler beim Verbinden der Module: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise

    def start(self, mode: str = None, auto_trade: bool = True):
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
            self.data_pipeline.start_auto_updates()
            self.module_status['data_pipeline']['status'] = "running"
            self.black_swan_detector.start_monitoring()
            self.module_status['black_swan_detector']['status'] = "running"
            self.telegram_interface.start()
            self.module_status['telegram_interface']['status'] = "running"
            
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
            
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Trading Bot erfolgreich gestartet")
            self._add_event("system", "Bot gestartet", {"mode": mode, "auto_trade": auto_trade})
            self._send_notification("Bot gestartet", f"Modus: {current_mode}\nTrading: {'Aktiviert' if auto_trade else 'Deaktiviert'}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstart", {"error": str(e)})
            return False

    def train_models(self):
        if not hasattr(self, 'learning_module'):
            self.logger.error("Learning Module nicht initialisiert")
            return False
        try:
            self.logger.info("Starte Modelltraining...")
            training_result = self.learning_module.train_all_models()
            self._add_event("learning", "Modelltraining durchgeführt", training_result)
            self.logger.info(f"Modelltraining abgeschlossen: {training_result}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Modelltraining: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._add_event("error", "Fehler beim Modelltraining", {"error": str(e)})
            return False

    def process_transcript(self, transcript_path: str):
        return self._process_transcript(transcript_path)

    def stop(self):
        if self.state not in [BotState.RUNNING, BotState.PAUSED]:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        try:
            self.logger.info("Stoppe Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.STOPPING
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "stopped"
            self.black_swan_detector.stop_monitoring()
            self.module_status['black_swan_detector']['status'] = "stopped"
            self.data_pipeline.stop_auto_updates()
            self.module_status['data_pipeline']['status'] = "stopped"
            self.running = False
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=10)
            self.state = BotState.READY
            self.logger.info("Trading Bot erfolgreich gestoppt")
            self._add_event("system", "Bot gestoppt", {})
            self._send_notification("Bot gestoppt", "Trading-Aktivitäten wurden beendet")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Stoppen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Botstopp", {"error": str(e)})
            return False

    def pause(self):
        if self.state != BotState.RUNNING:
            self.logger.warning(f"Bot ist nicht aktiv (Status: {self.state})")
            return False
        try:
            self.logger.info("Pausiere Trading Bot...")
            self.previous_state = self.state
            self.state = BotState.PAUSED
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "paused"
            self.logger.info("Trading Bot erfolgreich pausiert")
            self._add_event("system", "Bot pausiert", {})
            self._send_notification("Bot pausiert", "Trading wurde pausiert, Überwachung bleibt aktiv")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Pausieren des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Pausieren", {"error": str(e)})
            return False

    def resume(self):
        if self.state != BotState.PAUSED:
            self.logger.warning(f"Bot ist nicht pausiert (Status: {self.state})")
            return False
        try:
            self.logger.info("Setze Trading Bot fort...")
            self.previous_state = self.state
            self.state = BotState.RUNNING
            if self.module_status['live_trading']['status'] == "paused":
                self.live_trading.start_trading()  # Resume ohne mode-Parameter
                self.module_status['live_trading']['status'] = "running"
            self.logger.info("Trading Bot erfolgreich fortgesetzt")
            self._add_event("system", "Bot fortgesetzt", {})
            self._send_notification("Bot fortgesetzt", "Trading wurde wieder aktiviert")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Fortsetzen des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Fortsetzen", {"error": str(e)})
            return False

    def restart(self):
        try:
            self.logger.info("Starte Trading Bot neu...")
            success = self.stop()
            if not success:
                self.logger.error("Fehler beim Stoppen für Neustart")
                return False
            time.sleep(3)
            return self.start()
        except Exception as e:
            self.logger.error(f"Fehler beim Neustarten des Bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self._add_event("error", "Fehler beim Neustart", {"error": str(e)})
            return False

    def _main_loop(self):
        self.logger.info("Hauptschleife gestartet")
        while self.running:
            try:
                self._process_events()
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
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Fehler in der Hauptschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                self._add_event("error", "Fehler in Hauptschleife", {"error": str(e)})
                time.sleep(5)
        self.logger.info("Hauptschleife beendet")

    def _process_events(self):
        try:
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
                    self.event_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Fehler bei der Event-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _monitor_loop(self):
        self.logger.info("Überwachungsschleife gestartet")
        check_interval = 30
        health_check_counter = 0
        while self.running:
            try:
                for module_name, module in self.modules.items():
                    if hasattr(module, 'get_status'):
                        status = module.get_status()
                        if isinstance(status, dict):
                            self.module_status[module_name]['last_status'] = status
                            if 'error' in status and status.get('error'):
                                self.module_status[module_name]['errors'].append({
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'error': status.get('error')
                                })
                                self.logger.warning(f"Fehler in Modul {module_name}: {status.get('error')}")
                health_check_counter += 1
                if health_check_counter >= 5:
                    health_check_counter = 0
                    self._perform_health_check()
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Fehler in der Überwachungsschleife: {str(e)}")
                self.logger.error(traceback.format_exc())
                time.sleep(check_interval * 2)
        self.logger.info("Überwachungsschleife beendet")

    def _perform_health_check(self):
        self.logger.debug("Führe System-Health-Check durch...")
        try:
            for module_name, module in self.modules.items():
                if module_name == 'data_pipeline':
                    if hasattr(self.data_pipeline, 'get_last_update_time'):
                        last_update = self.data_pipeline.get_last_update_time('crypto')
                        if last_update:
                            time_diff = (datetime.datetime.now() - last_update).total_seconds()
                            if time_diff > 300:
                                self.logger.warning(f"Daten für 'crypto' sind veraltet ({time_diff:.0f} Sekunden)")
                elif module_name == 'live_trading':
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
        severity = event_data.get('severity', 0)
        title = event_data.get('title', 'Black Swan Event')
        message = event_data.get('message', 'Unbekanntes Marktereignis erkannt')
        details = event_data.get('details', {})
        self.logger.warning(f"Black Swan Event erkannt: {title} (Schweregrad: {severity:.2f})")
        self._add_event("black_swan", title, {"severity": severity, "message": message, "details": details})
        if severity > 0.8:
            self._emergency_shutdown(message)
        elif severity > 0.5:
            if self.state == BotState.RUNNING:
                self.pause()
            self._send_notification(f"⚠️ KRITISCHES MARKTEREIGNIS: {title}", message, priority="high")
        else:
            self._send_notification(f"⚠️ Ungewöhnliches Marktereignis: {title}", message)

    def _emergency_shutdown(self, reason: str):
        self.logger.critical(f"NOTFALL-SHUTDOWN eingeleitet: {reason}")
        try:
            self.previous_state = self.state
            self.state = BotState.EMERGENCY
            self.emergency_mode = True
            if (self.module_status['live_trading']['status'] == "running" and hasattr(self.live_trading, 'close_all_positions')):
                self.logger.critical("Schließe alle Positionen...")
                try:
                    result = self.live_trading.close_all_positions()
                    self.logger.info(f"Positionen geschlossen: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Schließen aller Positionen: {str(e)}")
            if (self.module_status['live_trading']['status'] == "running" and hasattr(self.live_trading, 'cancel_all_orders')):
                self.logger.critical("Storniere alle offenen Orders...")
                try:
                    result = self.live_trading.cancel_all_orders()
                    self.logger.info(f"Orders storniert: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Stornieren aller Orders: {str(e)}")
            if self.module_status['live_trading']['status'] == "running":
                self.live_trading.stop_trading()
                self.module_status['live_trading']['status'] = "emergency_stopped"
            self._send_notification("🚨 NOTFALL-SHUTDOWN AKTIVIERT 🚨",
                                    f"Grund: {reason}\nAlle Positionen wurden geschlossen und das Trading deaktiviert.",
                                    priority="critical")
            self._add_event("emergency", "Notfall-Shutdown", {"reason": reason})
        except Exception as e:
            self.logger.error(f"Fehler beim Notfall-Shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _handle_trading_error(self, error_data: Dict[str, Any]):
        message = error_data.get('message', 'Unbekannter Trading-Fehler')
        self.logger.error(f"Trading-Fehler: {message}")
        self._add_event("error", "Trading-Fehler", error_data)

    def _handle_order_update(self, order_data: Dict[str, Any]):
        self.logger.info(f"Order Update: {order_data}")

    def _handle_position_update(self, position_data: Dict[str, Any]):
        self.logger.info(f"Position Update: {position_data}")

    def _handle_error_event(self, error_data: Dict[str, Any]):
        self.logger.error(f"Fehler-Ereignis: {error_data}")

    def _handle_command_event(self, command_data: Dict[str, Any]):
        self.logger.info(f"Befehl-Ereignis: {command_data}")

    def _add_event(self, event_type: str, title: str, data: Dict[str, Any]):
        event = {
            "type": event_type,
            "title": title,
            "data": data,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def _send_notification(self, title: str, message: str, priority: str = "normal"):
        now = datetime.datetime.now()
        last_time = self.last_notification_time.get(priority, now - timedelta(seconds=self.notification_cooldown + 1))
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug(f"Notification für Priorität '{priority}' im Cooldown.")
            return
        self.last_notification_time[priority] = now
        notification_text = f"[{priority.upper()}] {title}\n{message}"
        self._send_notification_to_all(notification_text)

    def _send_notification_to_all(self, text: str):
        for user_id in self.allowed_users:
            try:
                self._send_message(int(user_id), text)
            except Exception as e:
                self.logger.error(f"Fehler beim Senden der Benachrichtigung an {user_id}: {str(e)}")

    def _send_main_menu(self, chat_id: int):
        reply_markup = {
            "inline_keyboard": [
                [{"text": "Start", "callback_data": "start_bot"}, {"text": "Stop", "callback_data": "stop_bot"}],
                [{"text": "Kontostand", "callback_data": "balance"}, {"text": "Positionen", "callback_data": "positions"}],
                [{"text": "Performance", "callback_data": "performance"}, {"text": "Report", "callback_data": "report"}],
                [{"text": "Notfall Stop", "callback_data": "confirm_emergency_stop"}]
            ]
        }
        self._send_message(chat_id, "Hauptmenü:", reply_markup)

    def process_callback_update(self, update: Dict[str, Any]):
        self._handle_callback_query(update)

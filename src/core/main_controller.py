import os
import sys
import logging
import threading
import time
import datetime
import traceback
import signal
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

# Modul-Imports – passe die Pfade an Deine Projektstruktur an!
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
    EMERGENCY = "emergency"

class MainController:
    """
    Hauptcontroller für den Trading Bot. Koordiniert alle Module und steuert den Betrieb.
    """
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.logger = logging.getLogger("MainController")
        self.logger.info("Initialisiere MainController...")
        self.config_manager = config_manager
        
        # Zustände und Flags
        self.state = BotState.INITIALIZING
        self.previous_state = None
        self.emergency_mode = False
        self.events = []
        self.max_events = 1000
        self.running = False
        self.shutdown_requested = False
        self.restart_requested = False
        self.pause_requested = False
        
        # Threads und Queue
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.event_queue = queue.Queue()
        self.main_thread = None
        self.monitor_thread = None
        
        # Signalhandler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Module
        self.modules: Dict[str, Any] = {}
        self.module_status: Dict[str, Any] = {}
        
        # Konfiguration laden
        if self.config_manager:
            self.config = self.config_manager.get_config()
            level = self.config.get('general', {}).get('log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, level))
            self.data_path = Path(self.config.get('general', {}).get('data_path', 'data'))
            self.data_path.mkdir(parents=True, exist_ok=True)
            self._initialize_modules()
            self.state = BotState.READY
            self.logger.info("MainController erfolgreich initialisiert")
        else:
            self.logger.error("Kein ConfigManager übergeben!")
            self.config = {}
            self.data_path = Path('data')
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.state = BotState.ERROR

    def _signal_handler(self, sig, frame):
        self.logger.info(f"Signal {sig} empfangen – Shutdown wird eingeleitet.")
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
            self.logger.error(f"Fehler bei Modulinitialisierung: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            raise

    def _connect_modules(self):
        try:
            self.black_swan_detector.register_notification_callback(self._handle_black_swan_event)
            # Telegram-Befehle registrieren
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

    def start(self, mode: str = None, auto_trade: bool = True) -> bool:
        if self.state == BotState.RUNNING:
            self.logger.info("Bot läuft bereits – Startbefehl wird ignoriert.")
            return True
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

            # Starte Live Trading falls aktiviert
            current_mode = mode or self.config.get('trading', {}).get('mode', 'paper')
            if auto_trade and current_mode != 'disabled':
                if getattr(self.live_trading, 'is_ready', False):
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

    def train_models(self) -> bool:
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

    def process_transcript(self, transcript_path: str) -> Dict[str, Any]:
        return self._process_transcript(transcript_path)

    def stop(self) -> bool:
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

    def pause(self) -> bool:
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

    def resume(self) -> bool:
        if self.state != BotState.PAUSED:
            self.logger.warning(f"Bot ist nicht pausiert (Status: {self.state})")
            return False
        try:
            self.logger.info("Setze Trading Bot fort...")
            self.previous_state = self.state
            self.state = BotState.RUNNING
            if self.module_status['live_trading']['status'] == "paused":
                self.live_trading.start_trading()
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

    def restart(self) -> bool:
        try:
            self.logger.info("Starte Trading Bot neu...")
            if not self.stop():
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
            if hasattr(self.data_pipeline, 'get_last_update_time'):
                last_update = self.data_pipeline.get_last_update_time('crypto')
                if last_update:
                    diff = (datetime.datetime.now() - last_update).total_seconds()
                    if diff > 300:
                        self.logger.warning(f"Daten für 'crypto' sind veraltet ({diff:.0f} Sekunden)")
            if self.module_status['live_trading']['status'] == "running" and hasattr(self.live_trading, 'get_status'):
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
        msg = event_data.get('message', 'Unbekanntes Marktereignis erkannt')
        details = event_data.get('details', {})
        self.logger.warning(f"Black Swan Event: {title} (Schweregrad: {severity:.2f})")
        self._add_event("black_swan", title, {"severity": severity, "message": msg, "details": details})
        if severity > 0.8:
            self._emergency_shutdown(msg)
        elif severity > 0.5:
            if self.state == BotState.RUNNING:
                self.pause()
            self._send_notification(f"⚠️ KRITISCHES MARKTEREIGNIS: {title}", msg, priority="high")
        else:
            self._send_notification(f"⚠️ Ungewöhnliches Marktereignis: {title}", msg)

    def _emergency_shutdown(self, reason: str):
        self.logger.critical(f"NOTFALL-SHUTDOWN: {reason}")
        try:
            self.previous_state = self.state
            self.state = BotState.EMERGENCY
            self.emergency_mode = True
            if self.module_status['live_trading']['status'] == "running" and hasattr(self.live_trading, 'close_all_positions'):
                self.logger.critical("Schließe alle Positionen...")
                try:
                    result = self.live_trading.close_all_positions()
                    self.logger.info(f"Positionen geschlossen: {result}")
                except Exception as e:
                    self.logger.error(f"Fehler beim Schließen aller Positionen: {str(e)}")
            if self.module_status['live_trading']['status'] == "running" and hasattr(self.live_trading, 'cancel_all_orders'):
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
                                    f"Grund: {reason}\nAlle Positionen geschlossen, Trading deaktiviert.",
                                    priority="critical")
            self._add_event("emergency", "Notfall-Shutdown", {"reason": reason})
        except Exception as e:
            self.logger.error(f"Fehler beim Notfall-Shutdown: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _handle_trading_error(self, error_data: Dict[str, Any]):
        msg = error_data.get('message', 'Unbekannter Trading-Fehler')
        context = error_data.get('context', '')
        errors = error_data.get('consecutive_errors', 0)
        self.logger.error(f"Trading-Fehler: {msg} (Kontext: {context})")
        self._add_event("error", "Trading-Fehler", {"message": msg, "context": context, "consecutive_errors": errors})
        if errors >= 5:
            self.logger.warning(f"Zu viele Fehler ({errors}), Trading wird pausiert")
            if self.state == BotState.RUNNING:
                self.pause()
            self._send_notification("🛑 Trading automatisch pausiert",
                                    f"{errors} Fehler in Folge.\nLetzter Fehler: {msg}",
                                    priority="high")

    def _handle_order_update(self, order_data: Dict[str, Any]):
        oid = order_data.get('id', 'unknown')
        symbol = order_data.get('symbol', 'unknown')
        status = order_data.get('status', 'unknown')
        self.logger.info(f"Order-Update: {oid} für {symbol} – Status: {status}")
        if hasattr(self.tax_module, 'process_order'):
            self.tax_module.process_order(order_data)
        if status == 'closed':
            side = order_data.get('side', 'unknown')
            amount = order_data.get('amount', 0)
            price = order_data.get('price', 0)
            cost = order_data.get('cost', 0)
            self._send_notification(f"Order ausgeführt: {symbol}",
                                    f"ID: {oid}\nTyp: {side}\nMenge: {amount}\nPreis: {price}\nWert: {cost}",
                                    priority="low")
        self._add_event("order", f"Order {status}", order_data)

    def _handle_position_update(self, position_data: Dict[str, Any]):
        symbol = position_data.get('symbol', 'unknown')
        action = position_data.get('action', 'unknown')
        self.logger.info(f"Positions-Update: {symbol} – Aktion: {action}")
        if action == 'close':
            side = position_data.get('side', 'unknown')
            contracts = position_data.get('contracts_before', 0)
            pnl = position_data.get('pnl', 0)
            pnl_pct = position_data.get('pnl_percent', 0)
            msg = f"Richtung: {side}\nKontrakte: {contracts}\nPnL: {pnl:.2f} ({pnl_pct:.2f}%)"
            title = f"Position {'mit Gewinn' if pnl > 0 else 'mit Verlust'} geschlossen: {symbol}"
            priority = "high" if pnl_pct < -5 else "normal"
            self._send_notification(title, msg, priority=priority)
        elif action == 'open':
            side = position_data.get('side', 'unknown')
            contracts = position_data.get('contracts', 0)
            entry = position_data.get('entry_price', 0)
            leverage = position_data.get('leverage', 1)
            self._send_notification(f"Neue Position eröffnet: {symbol}",
                                    f"Richtung: {side}\nKontrakte: {contracts}\nEinstieg: {entry}\nHebel: {leverage}x",
                                    priority="normal")
        self._add_event("position", f"Position {action}", position_data)

    def _handle_error_event(self, error_data: Dict[str, Any]):
        module = error_data.get('module', 'unknown')
        msg = error_data.get('message', 'Unbekannter Fehler')
        level = error_data.get('level', 'error')
        if level == 'critical':
            self.logger.critical(f"Kritischer Fehler in {module}: {msg}")
        else:
            self.logger.error(f"Fehler in {module}: {msg}")
        self._add_event("error", f"Fehler in {module}", error_data)
        if level == 'critical':
            self._send_notification(f"Kritischer Fehler in {module}", msg, priority="high")

    def _handle_command_event(self, command_data: Dict[str, Any]):
        command = command_data.get('command', '')
        params = command_data.get('params', {})
        source = command_data.get('source', 'unknown')
        self.logger.info(f"Kommando empfangen: {command} von {source}")
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
        symbol = trade_data.get('symbol', 'unknown')
        side = trade_data.get('side', 'unknown')
        price = trade_data.get('price', 0)
        amount = trade_data.get('amount', 0)
        self.logger.info(f"Trade: {symbol} {side} {amount} @ {price}")
        if hasattr(self.tax_module, 'process_trade'):
            self.tax_module.process_trade(trade_data)
        self._add_event("trade", "Trade ausgeführt", trade_data)

    def _send_notification(self, title: str, message: str, priority: str = "normal"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{message}\n\nZeit: {timestamp}"
        try:
            self.telegram_interface.send_notification(title, formatted, priority)
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Benachrichtigung: {str(e)}")

    def _add_event(self, event_type: str, title: str, data: Dict[str, Any]):
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': event_type,
            'title': title,
            'data': data
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def _get_account_balance(self) -> Dict[str, Any]:
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                hasattr(self.live_trading, 'get_account_balance')):
                balance = self.live_trading.get_account_balance()
                return {'status': 'success', 'balance': balance}
            else:
                return {'status': 'error', 'message': 'Live Trading ist nicht aktiv'}
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            return {'status': 'error', 'message': f"Fehler: {str(e)}"}

    def _get_open_positions(self) -> Dict[str, Any]:
        try:
            if (self.module_status['live_trading']['status'] in ["running", "paused"] and
                hasattr(self.live_trading, 'get_open_positions')):
                positions = self.live_trading.get_open_positions()
                return {'status': 'success', 'positions': positions}
            else:
                return {'status': 'error', 'message': 'Live Trading ist nicht aktiv'}
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen offener Positionen: {str(e)}")
            return {'status': 'error', 'message': f"Fehler: {str(e)}"}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        try:
            metrics = {}
            if hasattr(self.learning_module, 'performance_metrics'):
                metrics['learning'] = self.learning_module.performance_metrics
            if hasattr(self.learning_module, 'trade_history'):
                trades = self.learning_module.trade_history
                closed = [t for t in trades if getattr(t, 'status', None) == 'closed']
                if closed:
                    win = [t for t in closed if getattr(t, 'pnl_percent', 0) > 0]
                    lose = [t for t in closed if getattr(t, 'pnl_percent', 0) <= 0]
                    metrics['trading'] = {
                        'total_trades': len(closed),
                        'winning_trades': len(win),
                        'losing_trades': len(lose),
                        'win_rate': len(win) / len(closed),
                        'avg_win': sum(t.pnl_percent for t in win) / len(win) if win else 0,
                        'avg_loss': sum(t.pnl_percent for t in lose) / len(lose) if lose else 0,
                        'total_pnl': sum(t.pnl_percent for t in closed)
                    }
            if hasattr(self.tax_module, 'get_tax_summary'):
                metrics['tax'] = self.tax_module.get_tax_summary()
            return {'status': 'success', 'metrics': metrics}
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Performance-Metriken: {str(e)}")
            return {'status': 'error', 'message': f"Fehler: {str(e)}"}

    def _process_transcript(self, transcript_path: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Verarbeite Transkript: {transcript_path}")
            if not os.path.exists(transcript_path):
                return {'status': 'error', 'message': f"Datei nicht gefunden: {transcript_path}"}
            if hasattr(self.transcript_processor, 'process_transcript'):
                result = self.transcript_processor.process_transcript(transcript_path)
                self._add_event("transcript", "Transkript verarbeitet", {"path": transcript_path, "result": result})
                self._send_notification("Transkript verarbeitet", f"Datei: {transcript_path}\nErgebnis: {result.get('status', 'Unbekannt')}")
                return {'status': 'success', 'result': result}
            else:
                return {'status': 'error', 'message': 'TranscriptProcessor unterstützt process_transcript nicht'}
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            return {'status': 'error', 'message': f"Fehler: {str(e)}"}

    def _process_transcript_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        transcript_path = params.get('path', '')
        if not transcript_path:
            return {'status': 'error', 'message': 'Kein Transkript-Pfad angegeben'}
        return self._process_transcript(transcript_path)

    def get_status(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'previous_state': self.previous_state,
            'emergency_mode': self.emergency_mode,
            'running': self.running,
            'modules': self.module_status,
            'last_update': datetime.datetime.now().isoformat(),
            'events': self.events[-10:],
            'version': '1.0.0',
            'uptime': "00:00:00"
        }

    def _send_notification(self, title: str, message: str, priority: str = "normal"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"{message}\n\nZeit: {timestamp}"
        try:
            self.telegram_interface.send_notification(title, formatted_message, priority)
        except Exception as e:
            self.logger.error(f"Fehler beim Senden der Telegram-Benachrichtigung: {str(e)}")

if __name__ == "__main__":
    try:
        from src.core.config_manager import ConfigManager
        cm = ConfigManager()
        controller = MainController(cm)
        if controller.state == BotState.READY:
            controller.start(auto_trade=False)
            while True:
                time.sleep(1)
        else:
            print(f"Bot konnte nicht gestartet werden. Status: {controller.state}")
    except Exception as ex:
        print(f"Kritischer Fehler: {str(ex)}")
        traceback.print_exc()

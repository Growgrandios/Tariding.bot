# live_trading.py

import os
import sys
import time
import json
import hmac
import base64
import hashlib
import logging
import threading
import requests
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import ccxt
from dotenv import load_dotenv

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler()
    ]
)

class OrderType(Enum):
    """Mögliche Order-Typen."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Mögliche Order-Seiten (Kauf/Verkauf)."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Mögliche Order-Status."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"

class PositionSide(Enum):
    """Mögliche Positionsrichtungen."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Manche Börsen verwenden "both" für Hedged-Positionen

class LiveTradingConnector:
    """
    Verbindung zu Bitget für Echtzeit-Handel.
    Implementiert API-Aufrufe, Order-Management und Fehlerbehandlung.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Live Trading Connector.
        
        Args:
            config: Konfigurationseinstellungen mit API-Schlüsseln und Handelsparametern
        """
        self.logger = logging.getLogger("LiveTradingConnector")
        self.logger.info("Initialisiere Live Trading Connector...")
        
        # API-Konfiguration
        self.api_key = config.get('api_key', os.getenv('BITGET_API_KEY', ''))
        self.api_secret = config.get('api_secret', os.getenv('BITGET_API_SECRET', ''))
        self.api_passphrase = config.get('api_passphrase', os.getenv('BITGET_API_PASSPHRASE', ''))
        
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            self.logger.error("Bitget API-Schlüssel fehlen. Bitte in der Konfiguration oder .env-Datei angeben.")
            self.is_ready = False
        else:
            self.is_ready = True
        
        # Handelsparameter
        self.sandbox_mode = config.get('sandbox_mode', True)  # Standardmäßig im Sandbox-Modus starten
        self.default_leverage = config.get('default_leverage', 1)
        self.default_margin_mode = config.get('margin_mode', 'cross')  # 'cross' oder 'isolated'
        self.max_open_orders = config.get('max_open_orders', 50)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)  # Sekunden
        
        # Risikoparameter
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% des Kapitals
        self.max_leverage = config.get('max_leverage', 5)
        self.default_stop_loss_pct = config.get('default_stop_loss_pct', 0.05)  # 5%
        self.default_take_profit_pct = config.get('default_take_profit_pct', 0.1)  # 10%
        
        # Status und Fehlerbehandlung
        self.is_trading_active = False
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        self.exchange_status = "disconnected"
        
        # Cache für Marktdaten
        self.symbol_info_cache = {}
        self.ticker_cache = {}
        self.orderbook_cache = {}
        self.last_cache_update = datetime.now() - timedelta(hours=1)  # Cache ist initial abgelaufen
        self.cache_ttl = config.get('cache_ttl', 60)  # Sekunden
        
        # Ratenlimit-Tracking
        self.rate_limits = {
            "orders": {
                "limit": 20,  # Anfragen pro Sekunde
                "remaining": 20,
                "reset_time": datetime.now()
            },
            "market_data": {
                "limit": 50,  # Anfragen pro Sekunde
                "remaining": 50,
                "reset_time": datetime.now()
            }
        }
        
        # Callback-Funktionen
        self.order_update_callbacks = []
        self.position_update_callbacks = []
        self.error_callbacks = []
        
        # Initialisiere CCXT Exchange
        self._initialize_exchange()
        
        # Hintergrund-Tasks
        self.background_threads = {}
        
        self.logger.info("Live Trading Connector erfolgreich initialisiert")
    
    def _initialize_exchange(self):
        """Initialisiert die CCXT-Exchange-Instanz."""
        try:
            # Bitget-Verbindung über CCXT
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Default zum Perpetual Futures Handel
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            })
            
            # Sandbox-Modus (falls aktiviert)
            if self.sandbox_mode:
                self.exchange.set_sandbox_mode(True)
                self.logger.warning("SANDBOX-MODUS AKTIV! Es werden keine echten Trades durchgeführt.")
            
            # Verbindung testen
            if self.is_ready:
                self.exchange.load_markets()
                self.logger.info(f"Verbindung zu Bitget erfolgreich hergestellt. {len(self.exchange.markets)} Märkte verfügbar.")
                self.exchange_status = "connected"
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Börsenverbindung: {str(e)}")
            self.exchange_status = "error"
            self.is_ready = False
    
    def _handle_rate_limit(self, rate_limit_type: str):
        """
        Verwaltet das Rate-Limiting für API-Anfragen.
        
        Args:
            rate_limit_type: Typ der Rate-Limitierung ('orders' oder 'market_data')
        """
        rate_limit = self.rate_limits.get(rate_limit_type)
        
        if not rate_limit:
            return
        
        # Prüfe, ob das Ratenlimit zurückgesetzt werden muss
        now = datetime.now()
        if now >= rate_limit['reset_time']:
            # Ratenlimit zurücksetzen
            rate_limit['remaining'] = rate_limit['limit']
            rate_limit['reset_time'] = now + timedelta(seconds=1)
        
        # Warten, wenn keine Anfragen mehr übrig sind
        if rate_limit['remaining'] <= 0:
            sleep_time = (rate_limit['reset_time'] - now).total_seconds()
            if sleep_time > 0:
                self.logger.debug(f"Rate-Limit erreicht für {rate_limit_type}, warte {sleep_time:.2f} Sekunden")
                time.sleep(sleep_time)
                # Nach dem Warten zurücksetzen
                rate_limit['remaining'] = rate_limit['limit']
                rate_limit['reset_time'] = datetime.now() + timedelta(seconds=1)
        
        # Verbleibende Anfragen reduzieren
        rate_limit['remaining'] -= 1
    
    def _execute_with_retry(self, func, *args, rate_limit_type='market_data', **kwargs):
        """
        Führt eine Funktion mit automatischem Retry bei Fehlern aus.
        
        Args:
            func: Die auszuführende Funktion
            *args: Argumente für die Funktion
            rate_limit_type: Typ der Rate-Limitierung
            **kwargs: Keyword-Argumente für die Funktion
            
        Returns:
            Ergebnis der Funktion oder None bei Fehler
        """
        attempts = 0
        
        while attempts < self.max_retry_attempts:
            try:
                # Rate-Limit berücksichtigen
                self._handle_rate_limit(rate_limit_type)
                
                # Funktion ausführen
                result = func(*args, **kwargs)
                
                # Bei Erfolg Fehlertracking zurücksetzen
                self.consecutive_errors = 0
                return result
                
            except ccxt.NetworkError as e:
                attempts += 1
                self.consecutive_errors += 1
                self.last_error_time = datetime.now()
                
                self.logger.warning(f"Netzwerkfehler (Versuch {attempts}/{self.max_retry_attempts}): {str(e)}")
                
                if attempts < self.max_retry_attempts:
                    # Exponentielles Backoff
                    wait_time = self.retry_delay * (2 ** (attempts - 1))
                    time.sleep(wait_time)
                else:
                    self._handle_error(e, "Maximale Anzahl von Wiederholungsversuchen erreicht")
                    return None
            
            except ccxt.ExchangeError as e:
                # Börsenspezifische Fehler, die meistens nicht durch Wiederholung gelöst werden
                self.consecutive_errors += 1
                self.last_error_time = datetime.now()
                self._handle_error(e, "Börsenfehler")
                return None
                
            except Exception as e:
                self.consecutive_errors += 1
                self.last_error_time = datetime.now()
                self._handle_error(e, "Unerwarteter Fehler")
                return None
        
        return None
    
    def _handle_error(self, exception: Exception, context: str):
        """
        Verarbeitet Fehler und ruft Callbacks auf.
        
        Args:
            exception: Die aufgetretene Exception
            context: Kontext des Fehlers für bessere Nachvollziehbarkeit
        """
        self.error_count += 1
        error_msg = f"{context}: {str(exception)}"
        self.logger.error(error_msg)
        
        # Kritische Fehlerbehandlung
        if self.consecutive_errors >= 5:
            self.logger.critical(f"Zu viele aufeinanderfolgende Fehler ({self.consecutive_errors}). Trading wird vorübergehend deaktiviert.")
            self.is_trading_active = False
        
        # Fehler-Callbacks aufrufen
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'message': error_msg,
            'context': context,
            'consecutive_errors': self.consecutive_errors
        }
        
        for callback in self.error_callbacks:
            try:
                callback(error_data)
            except Exception as callback_error:
                self.logger.error(f"Fehler im Error-Callback: {str(callback_error)}")
    
    def start_trading(self):
        """Aktiviert den Handelsmodus und startet Hintergrund-Tasks."""
        if not self.is_ready:
            self.logger.error("Trading kann nicht gestartet werden - Connector ist nicht bereit.")
            return False
        
        self.is_trading_active = True
        
        # Hintergrund-Tasks starten
        self._start_background_tasks()
        
        self.logger.info("Live Trading aktiviert")
        return True
    
    def stop_trading(self):
        """Deaktiviert den Handelsmodus und stoppt Hintergrund-Tasks."""
        self.is_trading_active = False
        
        # Hintergrund-Tasks stoppen
        self._stop_background_tasks()
        
        self.logger.info("Live Trading deaktiviert")
        return True
    
    def _start_background_tasks(self):
        """Startet Hintergrund-Tasks für Statusaktualisierungen und Monitoring."""
        # Thread für regelmäßige Kontostandabfragen
        self.background_threads['account_monitor'] = threading.Thread(
            target=self._account_monitor_loop,
            daemon=True
        )
        
        # Thread für Orderbook-Caching
        self.background_threads['orderbook_cache'] = threading.Thread(
            target=self._orderbook_cache_loop,
            daemon=True
        )
        
        # Thread für offene Order-Überwachung
        self.background_threads['order_monitor'] = threading.Thread(
            target=self._order_monitor_loop,
            daemon=True
        )
        
        # Threads starten
        for thread_name, thread in self.background_threads.items():
            thread.start()
            self.logger.debug(f"Hintergrund-Thread '{thread_name}' gestartet")
    
    def _stop_background_tasks(self):
        """Stoppt alle Hintergrund-Tasks."""
        # Die Threads sind als Daemon-Threads konfiguriert und werden automatisch beendet
        self.background_threads = {}
        self.logger.debug("Alle Hintergrund-Threads gestoppt")
    
    def _account_monitor_loop(self):
        """Loop zur Überwachung des Kontostands und offener Positionen."""
        while self.is_trading_active:
            try:
                # Kontostand und Positionen abfragen (max. alle 5 Sekunden)
                self.get_account_balance()
                self.get_open_positions()
                
                # Warten bis zur nächsten Abfrage
                time.sleep(5)
            
            except Exception as e:
                self.logger.error(f"Fehler im Account-Monitor-Loop: {str(e)}")
                time.sleep(10)  # Längere Pause bei Fehlern
    
    def _orderbook_cache_loop(self):
        """Loop zum Caching von Orderbook-Daten für häufig gehandelte Symbole."""
        common_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']
        
        while self.is_trading_active:
            try:
                for symbol in common_symbols:
                    # Orderbook aktualisieren
                    orderbook = self._execute_with_retry(
                        self.exchange.fetch_order_book,
                        symbol,
                        limit=20
                    )
                    
                    if orderbook:
                        self.orderbook_cache[symbol] = {
                            'data': orderbook,
                            'timestamp': datetime.now()
                        }
                
                # Warten bis zur nächsten Aktualisierung
                time.sleep(2)
            
            except Exception as e:
                self.logger.error(f"Fehler im Orderbook-Cache-Loop: {str(e)}")
                time.sleep(5)  # Längere Pause bei Fehlern
    
    def _order_monitor_loop(self):
        """Loop zur Überwachung offener Orders."""
        while self.is_trading_active:
            try:
                # Offene Orders abfragen
                open_orders = self.get_open_orders()
                
                if open_orders:
                    self.logger.debug(f"Aktuell {len(open_orders)} offene Order(s)")
                
                # Warten bis zur nächsten Abfrage
                time.sleep(5)
            
            except Exception as e:
                self.logger.error(f"Fehler im Order-Monitor-Loop: {str(e)}")
                time.sleep(10)  # Längere Pause bei Fehlern
    
    # API-Methoden für Trading
    
    def get_exchange_info(self):
        """
        Ruft Börseninformationen wie Handelsregeln und Limits ab.
        
        Returns:
            Dictionary mit Börseninformationen
        """
        return self._execute_with_retry(self.exchange.fetch_markets)
    
    def get_symbol_info(self, symbol: str):
        """
        Ruft Informationen zu einem bestimmten Symbol ab.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            
        Returns:
            Dictionary mit Symbolinformationen
        """
        # Prüfen, ob Daten im Cache vorhanden sind
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # Lade Märkte, falls noch nicht geschehen
        if not self.exchange.markets:
            self.exchange.load_markets()
        
        # Symbol-Info abrufen
        symbol_info = self.exchange.market(symbol)
        
        # Im Cache speichern
        self.symbol_info_cache[symbol] = symbol_info
        
        return symbol_info
    
    def get_ticker(self, symbol: str):
        """
        Ruft den aktuellen Ticker für ein Symbol ab.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            
        Returns:
            Dictionary mit Ticker-Informationen
        """
        return self._execute_with_retry(self.exchange.fetch_ticker, symbol)
    
    def get_orderbook(self, symbol: str, limit: int = 20):
        """
        Ruft das Orderbook für ein Symbol ab.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            limit: Tiefe des Orderbooks
            
        Returns:
            Dictionary mit Orderbook-Daten
        """
        # Prüfen, ob aktuelle Daten im Cache vorhanden sind
        now = datetime.now()
        if (symbol in self.orderbook_cache and 
            (now - self.orderbook_cache[symbol]['timestamp']).total_seconds() < self.cache_ttl):
            return self.orderbook_cache[symbol]['data']
        
        # Orderbook von der Börse abrufen
        orderbook = self._execute_with_retry(self.exchange.fetch_order_book, symbol, limit=limit)
        
        # Im Cache speichern
        if orderbook:
            self.orderbook_cache[symbol] = {
                'data': orderbook,
                'timestamp': now
            }
        
        return orderbook
    
    def get_account_balance(self):
        """
        Ruft den Kontostand und verfügbare Mittel ab.
        
        Returns:
            Dictionary mit Kontoinformationen
        """
        balance = self._execute_with_retry(self.exchange.fetch_balance, {'type': 'swap'}, rate_limit_type='orders')
        
        if balance:
            # Für Logging und Debugging
            total_usdt = balance.get('total', {}).get('USDT', 0)
            free_usdt = balance.get('free', {}).get('USDT', 0)
            self.logger.debug(f"Kontostand: {total_usdt} USDT (Verfügbar: {free_usdt} USDT)")
        
        return balance
    
    def get_open_positions(self):
        """
        Ruft alle offenen Positionen ab.
        
        Returns:
            Liste mit offenen Positionsdaten
        """
        positions = self._execute_with_retry(self.exchange.fetch_positions, None, rate_limit_type='orders')
        
        # Nur Positionen mit Größe != 0 behalten
        if positions:
            active_positions = [p for p in positions if float(p['contracts']) > 0]
            
            if active_positions:
                self.logger.debug(f"Offene Positionen: {len(active_positions)}")
                for pos in active_positions:
                    self.logger.debug(f"  {pos['symbol']}: {pos['side']} {pos['contracts']} Kontrakte, PnL: {pos['unrealizedPnl']}")
            
            return active_positions
        
        return []
    
    def get_open_orders(self, symbol: Optional[str] = None):
        """
        Ruft alle offenen Orders ab.
        
        Args:
            symbol: Optionales Symbol, um nur Orders für dieses Symbol abzurufen
            
        Returns:
            Liste mit offenen Orders
        """
        if symbol:
            return self._execute_with_retry(self.exchange.fetch_open_orders, symbol, rate_limit_type='orders')
        else:
            return self._execute_with_retry(self.exchange.fetch_open_orders, None, rate_limit_type='orders')
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50):
        """
        Ruft die Orderhistorie ab.
        
        Args:
            symbol: Optionales Symbol, um nur Orders für dieses Symbol abzurufen
            limit: Maximale Anzahl der abzurufenden Orders
            
        Returns:
            Liste mit historischen Orders
        """
        params = {'limit': limit}
        
        if symbol:
            return self._execute_with_retry(self.exchange.fetch_orders, symbol, params=params, rate_limit_type='orders')
        else:
            # Bei manchen Börsen ist ein Symbol erforderlich
            try:
                return self._execute_with_retry(self.exchange.fetch_orders, None, params=params, rate_limit_type='orders')
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen der Orderhistorie ohne Symbol: {str(e)}")
                return []
    
    def create_market_order(self, symbol: str, side: str, amount: float, 
                          reduce_only: bool = False, params: Dict = None):
        """
        Erstellt eine Market-Order.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            side: 'buy' oder 'sell'
            amount: Menge in Kontrakten oder USD (je nach Börse)
            reduce_only: Ob die Order nur zum Reduzieren einer Position verwendet werden soll
            params: Zusätzliche Parameter für die Order
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Market-Order wurde nicht erstellt.")
            return None
        
        if not params:
            params = {}
        
        if reduce_only:
            params['reduceOnly'] = True
        
        try:
            # Für Logging
            order_description = f"{side.upper()} {amount} {symbol} zum Marktpreis"
            self.logger.info(f"Erstelle Market-Order: {order_description}")
            
            # Order erstellen
            order = self._execute_with_retry(
                self.exchange.create_market_order,
                symbol, side, amount, None, params,
                rate_limit_type='orders'
            )
            
            if order:
                self.logger.info(f"Market-Order erfolgreich erstellt: {order['id']}")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.order_update_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error(f"Fehler in Order-Update-Callback: {str(e)}")
            
            return order
            
        except Exception as e:
            error_msg = f"Fehler beim Erstellen der Market-Order ({order_description}): {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float,
                         post_only: bool = False, reduce_only: bool = False, params: Dict = None):
        """
        Erstellt eine Limit-Order.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            side: 'buy' oder 'sell'
            amount: Menge in Kontrakten oder USD (je nach Börse)
            price: Limitpreis
            post_only: Ob die Order nur als Maker ausgeführt werden soll
            reduce_only: Ob die Order nur zum Reduzieren einer Position verwendet werden soll
            params: Zusätzliche Parameter für die Order
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Limit-Order wurde nicht erstellt.")
            return None
        
        if not params:
            params = {}
        
        if post_only:
            params['postOnly'] = True
        
        if reduce_only:
            params['reduceOnly'] = True
        
        try:
            # Für Logging
            order_description = f"{side.upper()} {amount} {symbol} zum Limit-Preis {price}"
            self.logger.info(f"Erstelle Limit-Order: {order_description}")
            
            # Order erstellen
            order = self._execute_with_retry(
                self.exchange.create_limit_order,
                symbol, side, amount, price, params,
                rate_limit_type='orders'
            )
            
            if order:
                self.logger.info(f"Limit-Order erfolgreich erstellt: {order['id']}")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.order_update_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error(f"Fehler in Order-Update-Callback: {str(e)}")
            
            return order
            
        except Exception as e:
            error_msg = f"Fehler beim Erstellen der Limit-Order ({order_description}): {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def create_stop_loss_order(self, symbol: str, side: str, amount: float, 
                             stop_price: float, params: Dict = None):
        """
        Erstellt eine Stop-Loss-Order.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            side: 'buy' oder 'sell'
            amount: Menge in Kontrakten oder USD (je nach Börse)
            stop_price: Auslösepreis für den Stop Loss
            params: Zusätzliche Parameter für die Order
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Stop-Loss-Order wurde nicht erstellt.")
            return None
        
        if not params:
            params = {}
        
        # Stop-Loss-Parameter anpassen (je nach Börse unterschiedlich)
        params['stopPrice'] = stop_price
        params['reduceOnly'] = True  # Stop-Loss sollte immer reduce-only sein
        
        try:
            # Für Logging
            order_description = f"{side.upper()} {amount} {symbol} bei Stop-Loss {stop_price}"
            self.logger.info(f"Erstelle Stop-Loss-Order: {order_description}")
            
            # Order erstellen
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'stop_market', side, amount, None, params,
                rate_limit_type='orders'
            )
            
            if order:
                self.logger.info(f"Stop-Loss-Order erfolgreich erstellt: {order['id']}")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.order_update_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error(f"Fehler in Order-Update-Callback: {str(e)}")
            
            return order
            
        except Exception as e:
            error_msg = f"Fehler beim Erstellen der Stop-Loss-Order ({order_description}): {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def create_take_profit_order(self, symbol: str, side: str, amount: float, 
                               take_profit_price: float, params: Dict = None):
        """
        Erstellt eine Take-Profit-Order.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT:USDT')
            side: 'buy' oder 'sell'
            amount: Menge in Kontrakten oder USD (je nach Börse)
            take_profit_price: Auslösepreis für den Take Profit
            params: Zusätzliche Parameter für die Order
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Take-Profit-Order wurde nicht erstellt.")
            return None
        
        if not params:
            params = {}
        
        # Take-Profit-Parameter anpassen (je nach Börse unterschiedlich)
        params['stopPrice'] = take_profit_price
        params['reduceOnly'] = True  # Take-Profit sollte immer reduce-only sein
        
        try:
            # Für Logging
            order_description = f"{side.upper()} {amount} {symbol} bei Take-Profit {take_profit_price}"
            self.logger.info(f"Erstelle Take-Profit-Order: {order_description}")
            
            # Order erstellen (für Bitget wird 'conditional' verwendet)
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'take_profit_market', side, amount, None, params,
                rate_limit_type='orders'
            )
            
            if order:
                self.logger.info(f"Take-Profit-Order erfolgreich erstellt: {order['id']}")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.order_update_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        self.logger.error(f"Fehler in Order-Update-Callback: {str(e)}")
            
            return order
            
        except Exception as e:
            error_msg = f"Fehler beim Erstellen der Take-Profit-Order ({order_description}): {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def cancel_order(self, order_id: str, symbol: str):
        """
        Storniert eine Order.
        
        Args:
            order_id: ID der zu stornierenden Order
            symbol: Handelssymbol der Order
            
        Returns:
            Stornierungsbestätigung oder None bei Fehler
        """
        try:
            self.logger.info(f"Storniere Order {order_id} für {symbol}")
            
            result = self._execute_with_retry(
                self.exchange.cancel_order,
                order_id, symbol,
                rate_limit_type='orders'
            )
            
            if result:
                self.logger.info(f"Order {order_id} erfolgreich storniert")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.order_update_callbacks:
                    try:
                        # Für Callbacks ein Update mit Status "canceled" erstellen
                        update_data = {
                            'id': order_id,
                            'symbol': symbol,
                            'status': 'canceled',
                            'timestamp': datetime.now().timestamp() * 1000
                        }
                        callback(update_data)
                    except Exception as e:
                        self.logger.error(f"Fehler in Order-Update-Callback: {str(e)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Fehler beim Stornieren der Order {order_id}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """
        Storniert alle offenen Orders.
        
        Args:
            symbol: Optionales Symbol, um nur Orders für dieses Symbol zu stornieren
            
        Returns:
            Anzahl der stornierten Orders oder None bei Fehler
        """
        try:
            if symbol:
                self.logger.info(f"Storniere alle Orders für {symbol}")
                result = self._execute_with_retry(
                    self.exchange.cancel_all_orders,
                    symbol,
                    rate_limit_type='orders'
                )
            else:
                self.logger.info("Storniere alle Orders für alle Symbole")
                # Bei einigen Börsen muss man alle Symbole einzeln durchgehen
                open_orders = self.get_open_orders()
                
                if not open_orders:
                    return {'canceled': 0}
                
                # Gruppiere Orders nach Symbol
                orders_by_symbol = {}
                for order in open_orders:
                    symbol = order['symbol']
                    if symbol not in orders_by_symbol:
                        orders_by_symbol[symbol] = []
                    orders_by_symbol[symbol].append(order)
                
                # Storniere Orders für jedes Symbol
                canceled_count = 0
                for sym, orders in orders_by_symbol.items():
                    try:
                        self._execute_with_retry(
                            self.exchange.cancel_all_orders,
                            sym,
                            rate_limit_type='orders'
                        )
                        canceled_count += len(orders)
                    except Exception as sym_error:
                        self.logger.error(f"Fehler beim Stornieren aller Orders für {sym}: {str(sym_error)}")
                
                result = {'canceled': canceled_count}
            
            if result:
                self.logger.info(f"Stornierung abgeschlossen: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"Fehler beim Stornieren aller Orders: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def get_order_status(self, order_id: str, symbol: str):
        """
        Ruft den Status einer Order ab.
        
        Args:
            order_id: ID der Order
            symbol: Handelssymbol der Order
            
        Returns:
            Order-Status oder None bei Fehler
        """
        return self._execute_with_retry(
            self.exchange.fetch_order,
            order_id, symbol,
            rate_limit_type='orders'
        )
    
    def set_leverage(self, symbol: str, leverage: int):
        """
        Setzt den Hebel für ein Symbol.
        
        Args:
            symbol: Handelssymbol
            leverage: Hebel (z.B. 5 für 5x)
            
        Returns:
            Antwort der Börse oder None bei Fehler
        """
        try:
            self.logger.info(f"Setze Hebel auf {leverage}x für {symbol}")
            
            # Begrenze Hebel auf den maximalen erlaubten Wert
            if leverage > self.max_leverage:
                self.logger.warning(f"Hebel {leverage}x übersteigt Maximum von {self.max_leverage}x. Setze auf {self.max_leverage}x")
                leverage = self.max_leverage
            
            result = self._execute_with_retry(
                self.exchange.set_leverage,
                leverage, symbol,
                rate_limit_type='orders'
            )
            
            if result:
                self.logger.info(f"Hebel für {symbol} erfolgreich auf {leverage}x gesetzt")
            
            return result
            
        except Exception as e:
            error_msg = f"Fehler beim Setzen des Hebels für {symbol}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def set_margin_mode(self, symbol: str, margin_mode: str):
        """
        Setzt den Margin-Modus für ein Symbol.
        
        Args:
            symbol: Handelssymbol
            margin_mode: 'cross' oder 'isolated'
            
        Returns:
            Antwort der Börse oder None bei Fehler
        """
        if margin_mode not in ['cross', 'isolated']:
            self.logger.error(f"Ungültiger Margin-Modus: {margin_mode}. Muss 'cross' oder 'isolated' sein.")
            return None
        
        try:
            self.logger.info(f"Setze Margin-Modus auf {margin_mode} für {symbol}")
            
            result = self._execute_with_retry(
                self.exchange.set_margin_mode,
                margin_mode, symbol,
                rate_limit_type='orders'
            )
            
            if result:
                self.logger.info(f"Margin-Modus für {symbol} erfolgreich auf {margin_mode} gesetzt")
            
            return result
            
        except Exception as e:
            # Spezialfall: Manche Börsen werfen einen Fehler, wenn der Modus bereits gesetzt ist
            if "already" in str(e).lower():
                self.logger.info(f"Margin-Modus für {symbol} ist bereits {margin_mode}")
                return {'info': f"Margin mode is already set to {margin_mode}"}
            
            error_msg = f"Fehler beim Setzen des Margin-Modus für {symbol}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float,
                              risk_percentage: float, account_balance: Optional[float] = None):
        """
        Berechnet die optimale Positionsgröße basierend auf dem Risikomanagement.
        
        Args:
            symbol: Handelssymbol
            entry_price: Geplanter Einstiegspreis
            stop_loss_price: Geplanter Stop-Loss-Preis
            risk_percentage: Risiko als Prozentsatz des Kontostands (z.B. 0.01 für 1%)
            account_balance: Optionaler Kontostand (wenn nicht angegeben, wird er abgerufen)
            
        Returns:
            Berechnete Positionsgröße in Kontrakten oder None bei Fehler
        """
        try:
            # Kontostand abrufen, wenn nicht angegeben
            if account_balance is None:
                balance = self.get_account_balance()
                if not balance:
                    return None
                
                account_balance = balance.get('total', {}).get('USDT', 0)
            
            # Symbolinformationen abrufen
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Berechnung des zu riskierenden Betrags
            risk_amount = account_balance * risk_percentage
            
            # Berechnung des Risikos pro Kontrakt
            price_difference = abs(entry_price - stop_loss_price)
            risk_per_contract = price_difference * symbol_info.get('contractSize', 1)
            
            # Berechnung der Positionsgröße in Kontrakten
            position_size_contracts = risk_amount / risk_per_contract
            
            # Minimum und Maximum basierend auf Börsenregeln
            min_amount = symbol_info.get('limits', {}).get('amount', {}).get('min', 0)
            max_amount = symbol_info.get('limits', {}).get('amount', {}).get('max', float('inf'))
            
            # Anpassen an die Limits
            position_size_contracts = max(min_amount, min(position_size_contracts, max_amount))
            
            # Anpassen an Schrittweitenlimit
            step_size = symbol_info.get('precision', {}).get('amount', 0)
            if step_size > 0:
                position_size_contracts = self._round_to_precision(position_size_contracts, step_size)
            
            self.logger.info(
                f"Berechnete Positionsgröße für {symbol}: {position_size_contracts} Kontrakte "
                f"(Risiko: {risk_percentage*100}%, ${risk_amount:.2f})"
            )
            
            return position_size_contracts
            
        except Exception as e:
            error_msg = f"Fehler bei der Berechnung der Positionsgröße für {symbol}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def _round_to_precision(self, value: float, precision: float) -> float:
        """
        Rundet einen Wert zur angegebenen Präzision.
        
        Args:
            value: Zu rundender Wert
            precision: Präzision (Schrittweite)
            
        Returns:
            Gerundeter Wert
        """
        # Wenn die Präzision eine ganze Zahl ist (z.B. 1, 10, 100), verwenden wir sie direkt
        if precision.is_integer():
            precision = int(precision)
            return int(value / precision) * precision
        
        # Ansonsten ermitteln wir die Anzahl der Dezimalstellen
        str_precision = str(precision)
        decimal_places = len(str_precision.split('.')[-1])
        
        # Runden auf die entsprechende Anzahl von Dezimalstellen
        return round(value, decimal_places)
    
    def execute_trade_with_risk_management(self, symbol: str, side: str, entry_type: str,
                                         risk_percentage: float, stop_loss_pct: Optional[float] = None,
                                         take_profit_pct: Optional[float] = None, 
                                         entry_price: Optional[float] = None, leverage: int = 1):
        """
        Führt einen Trade mit vollem Risikomanagement aus.
        
        Args:
            symbol: Handelssymbol
            side: 'buy' oder 'sell'
            entry_type: 'market' oder 'limit'
            risk_percentage: Risiko als Prozentsatz des Kontostands
            stop_loss_pct: Prozentsatz für Stop-Loss (relativ zum Einstiegspreis)
            take_profit_pct: Prozentsatz für Take-Profit (relativ zum Einstiegspreis)
            entry_price: Einstiegspreis für Limit-Orders
            leverage: Zu verwendender Hebel
            
        Returns:
            Dictionary mit Trade-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Trade wurde nicht ausgeführt.")
            return None
        
        try:
            # Ticker abrufen für den aktuellen Preis
            ticker = self.get_ticker(symbol)
            if not ticker:
                return None
            
            current_price = ticker['last']
            
            # Einstiegspreis setzen
            if entry_type == 'market':
                actual_entry_price = current_price
            else:  # limit
                if not entry_price:
                    self.logger.error("Für Limit-Orders muss ein Einstiegspreis angegeben werden")
                    return None
                actual_entry_price = entry_price
            
            # Stop-Loss und Take-Profit berechnen
            if stop_loss_pct is None:
                stop_loss_pct = self.default_stop_loss_pct
            
            if take_profit_pct is None:
                take_profit_pct = self.default_take_profit_pct
            
            # Stop-Loss-Preis berechnen
            if side == 'buy':
                stop_loss_price = actual_entry_price * (1 - stop_loss_pct)
                take_profit_price = actual_entry_price * (1 + take_profit_pct)
            else:  # sell
                stop_loss_price = actual_entry_price * (1 + stop_loss_pct)
                take_profit_price = actual_entry_price * (1 - take_profit_pct)
            
            # Leverage setzen
            self.set_leverage(symbol, leverage)
            
            # Positionsgröße berechnen
            position_size = self.calculate_position_size(
                symbol, actual_entry_price, stop_loss_price, risk_percentage
            )
            
            if not position_size or position_size <= 0:
                self.logger.error(f"Ungültige Positionsgröße: {position_size}")
                return None
            
            # Order ausführen
            if entry_type == 'market':
                entry_order = self.create_market_order(symbol, side, position_size)
                if not entry_order:
                    return None
                
                # Tatsächlichen Ausführungspreis aus der Order nehmen
                if 'average' in entry_order and entry_order['average']:
                    actual_entry_price = entry_order['average']
                elif 'price' in entry_order and entry_order['price']:
                    actual_entry_price = entry_order['price']
                
                # Stop-Loss und Take-Profit neu berechnen, falls sich der Ausführungspreis geändert hat
                if side == 'buy':
                    stop_loss_price = actual_entry_price * (1 - stop_loss_pct)
                    take_profit_price = actual_entry_price * (1 + take_profit_pct)
                else:  # sell
                    stop_loss_price = actual_entry_price * (1 + stop_loss_pct)
                    take_profit_price = actual_entry_price * (1 - take_profit_pct)
                
            else:  # limit
                entry_order = self.create_limit_order(symbol, side, position_size, actual_entry_price)
                if not entry_order:
                    return None
            
            # Stop-Loss und Take-Profit Orders erstellen
            stop_loss_side = 'sell' if side == 'buy' else 'buy'
            take_profit_side = 'sell' if side == 'buy' else 'buy'
            
            # Bei Limit-Orders SL und TP erst nach Ausführung hinzufügen
            if entry_type == 'limit':
                self.logger.info(
                    f"Limit-Order erstellt. Stop-Loss und Take-Profit werden nach Ausführung platziert. "
                    f"Geplant: SL bei {stop_loss_price:.2f}, TP bei {take_profit_price:.2f}"
                )
                
                # Hier könnte ein separater Thread oder eine Callback-Funktion die Order überwachen
                # und SL/TP hinzufügen, sobald die Limit-Order ausgeführt wurde
                
                return {
                    'entry_order': entry_order,
                    'stop_loss': None,
                    'take_profit': None,
                    'planned_stop_loss': stop_loss_price,
                    'planned_take_profit': take_profit_price
                }
            
            # Bei Market-Orders sofort SL und TP hinzufügen
            stop_loss_order = self.create_stop_loss_order(
                symbol, stop_loss_side, position_size, stop_loss_price
            )
            
            take_profit_order = self.create_take_profit_order(
                symbol, take_profit_side, position_size, take_profit_price
            )
            
            return {
                'entry_order': entry_order,
                'stop_loss': stop_loss_order,
                'take_profit': take_profit_order,
                'entry_price': actual_entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'position_size': position_size
            }
            
        except Exception as e:
            error_msg = f"Fehler bei der Ausführung des Trades für {symbol}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def close_position(self, symbol: str, reduce_percentage: float = 1.0):
        """
        Schließt eine offene Position.
        
        Args:
            symbol: Handelssymbol
            reduce_percentage: Prozentsatz der Position, der geschlossen werden soll (1.0 = 100%)
            
        Returns:
            Order-Informationen oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Position wurde nicht geschlossen.")
            return None
        
        try:
            # Offene Positionen abrufen
            positions = self.get_open_positions()
            if not positions:
                self.logger.warning(f"Keine offene Position für {symbol} gefunden")
                return None
            
            # Position für das Symbol finden
            position = None
            for pos in positions:
                if pos['symbol'] == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.warning(f"Keine offene Position für {symbol} gefunden")
                return None
            
            # Informationen zur Position
            contracts = float(position['contracts'])
            side = position['side']
            
            if contracts <= 0:
                self.logger.warning(f"Position für {symbol} hat keine Kontrakte")
                return None
            
            # Gegengesetzte Seite für das Schließen
            close_side = 'sell' if side == 'long' else 'buy'
            
            # Zu schließende Kontraktmenge berechnen
            close_amount = contracts * reduce_percentage
            
            # Symbolinformationen für Mindeststückzahl
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                min_amount = symbol_info.get('limits', {}).get('amount', {}).get('min', 0)
                if close_amount < min_amount:
                    self.logger.warning(
                        f"Zu schließende Menge {close_amount} ist kleiner als Minimum {min_amount}. "
                        f"Verwende Minimum."
                    )
                    close_amount = min_amount
            
            # Market-Order zum Schließen erstellen
            self.logger.info(f"Schließe {reduce_percentage*100}% der {side} Position in {symbol}: {close_amount} Kontrakte")
            
            result = self.create_market_order(
                symbol, close_side, close_amount, reduce_only=True
            )
            
            if result:
                self.logger.info(f"Position in {symbol} erfolgreich geschlossen: {result['id']}")
                
                # Order-Update-Callbacks aufrufen
                for callback in self.position_update_callbacks:
                    try:
                        position_update = {
                            'symbol': symbol,
                            'side': side,
                            'contracts_before': contracts,
                            'contracts_after': contracts * (1 - reduce_percentage),
                            'action': 'reduce' if reduce_percentage < 1.0 else 'close',
                            'timestamp': datetime.now().timestamp() * 1000
                        }
                        callback(position_update)
                    except Exception as e:
                        self.logger.error(f"Fehler in Position-Update-Callback: {str(e)}")
            
            return result
            
        except Exception as e:
            error_msg = f"Fehler beim Schließen der Position für {symbol}: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def close_all_positions(self):
        """
        Schließt alle offenen Positionen.
        
        Returns:
            Dictionary mit Ergebnissen je Symbol oder None bei Fehler
        """
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Positionen wurden nicht geschlossen.")
            return None
        
        try:
            # Offene Positionen abrufen
            positions = self.get_open_positions()
            if not positions:
                self.logger.info("Keine offenen Positionen vorhanden")
                return {'closed': 0}
            
            # Alle Positionen schließen
            results = {}
            closed_count = 0
            
            for position in positions:
                symbol = position['symbol']
                result = self.close_position(symbol)
                
                if result:
                    results[symbol] = result
                    closed_count += 1
            
            results['closed'] = closed_count
            self.logger.info(f"{closed_count} von {len(positions)} Positionen geschlossen")
            
            return results
            
        except Exception as e:
            error_msg = "Fehler beim Schließen aller Positionen: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    # Callback-Registrierung
    
    def register_order_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Registriert eine Callback-Funktion für Order-Updates.
        
        Args:
            callback: Funktion, die bei Order-Updates aufgerufen wird
        """
        self.order_update_callbacks.append(callback)
        self.logger.debug("Order-Update-Callback registriert")
    
    def register_position_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Registriert eine Callback-Funktion für Positions-Updates.
        
        Args:
            callback: Funktion, die bei Positions-Updates aufgerufen wird
        """
        self.position_update_callbacks.append(callback)
        self.logger.debug("Position-Update-Callback registriert")
    
    def register_error_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Registriert eine Callback-Funktion für Fehler.
        
        Args:
            callback: Funktion, die bei Fehlern aufgerufen wird
        """
        self.error_callbacks.append(callback)
        self.logger.debug("Error-Callback registriert")
    
    # Status-Methoden
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Status des Trading-Connectors zurück.
        
        Returns:
            Status-Dictionary
        """
        return {
            'is_ready': self.is_ready,
            'is_trading_active': self.is_trading_active,
            'exchange_status': self.exchange_status,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'sandbox_mode': self.sandbox_mode
        }

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration für den Connector
    config = {
        'api_key': os.getenv('BITGET_API_KEY', ''),
        'api_secret': os.getenv('BITGET_API_SECRET', ''),
        'api_passphrase': os.getenv('BITGET_API_PASSPHRASE', ''),
        'sandbox_mode': True,  # Wichtig: Im Test immer Sandbox verwenden!
        'default_leverage': 3,
        'max_leverage': 10
    }
    
    # Connector initialisieren
    connector = LiveTradingConnector(config)
    
    # Status ausgeben
    status = connector.get_status()
    print(f"Connector Status: {status}")
    
    # Beispiel für Order-Update-Callback
    def on_order_update(order_data):
        print(f"Order Update: {order_data}")
    
    connector.register_order_update_callback(on_order_update)
    
    # Trading aktivieren
    if connector.is_ready:
        connector.start_trading()
        
        # Beispiel für einen Trade mit Risikomanagement
        # VORSICHT: Dieser Code würde einen echten Trade ausführen, wenn sandbox_mode=False wäre!
        """
        trade_result = connector.execute_trade_with_risk_management(
            symbol='BTC/USDT:USDT',
            side='buy',
            entry_type='market',
            risk_percentage=0.01,  # 1% des Kapitals
            stop_loss_pct=0.02,    # 2% Stop-Loss
            take_profit_pct=0.04,  # 4% Take-Profit
            leverage=3
        )
        
        print(f"Trade Ergebnis: {trade_result}")
        """
        
        # Trading wieder deaktivieren
        connector.stop_trading()

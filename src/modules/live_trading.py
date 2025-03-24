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
        self.sandbox_mode = config.get('sandbox_mode', True)
        self.default_leverage = config.get('default_leverage', 1)
        self.default_margin_mode = config.get('margin_mode', 'cross')
        self.max_open_orders = config.get('max_open_orders', 50)
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Risikoparameter
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_leverage = config.get('max_leverage', 5)
        self.default_stop_loss_pct = config.get('default_stop_loss_pct', 0.05)
        self.default_take_profit_pct = config.get('default_take_profit_pct', 0.1)
        
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
        self.last_cache_update = datetime.now() - timedelta(hours=1)
        self.cache_ttl = config.get('cache_ttl', 60)
        
        # Ratenlimit-Tracking
        self.rate_limits = {
            "orders": {"limit": 20, "remaining": 20, "reset_time": datetime.now()},
            "market_data": {"limit": 50, "remaining": 50, "reset_time": datetime.now()}
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
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            })
            if self.sandbox_mode:
                self.exchange.set_sandbox_mode(True)
                self.logger.warning("SANDBOX-MODUS AKTIV! Es werden keine echten Trades durchgeführt.")
            if self.is_ready:
                self.exchange.load_markets()
                self.logger.info(f"Verbindung zu Bitget erfolgreich hergestellt. {len(self.exchange.markets)} Märkte verfügbar.")
                self.exchange_status = "connected"
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Börsenverbindung: {str(e)}")
            self.exchange_status = "error"
            self.is_ready = False
    
    def _handle_rate_limit(self, rate_limit_type: str):
        rate_limit = self.rate_limits.get(rate_limit_type)
        if not rate_limit:
            return
        now = datetime.now()
        if now >= rate_limit['reset_time']:
            rate_limit['remaining'] = rate_limit['limit']
            rate_limit['reset_time'] = now + timedelta(seconds=1)
        if rate_limit['remaining'] <= 0:
            sleep_time = (rate_limit['reset_time'] - now).total_seconds()
            if sleep_time > 0:
                self.logger.debug(f"Rate-Limit erreicht für {rate_limit_type}, warte {sleep_time:.2f} Sekunden")
                time.sleep(sleep_time)
                rate_limit['remaining'] = rate_limit['limit']
                rate_limit['reset_time'] = datetime.now() + timedelta(seconds=1)
        rate_limit['remaining'] -= 1
    
    def _execute_with_retry(self, func, *args, rate_limit_type='market_data', **kwargs):
        attempts = 0
        while attempts < self.max_retry_attempts:
            try:
                self._handle_rate_limit(rate_limit_type)
                result = func(*args, **kwargs)
                self.consecutive_errors = 0
                return result
            except ccxt.NetworkError as e:
                attempts += 1
                self.consecutive_errors += 1
                self.last_error_time = datetime.now()
                self.logger.warning(f"Netzwerkfehler (Versuch {attempts}/{self.max_retry_attempts}): {str(e)}")
                if attempts < self.max_retry_attempts:
                    wait_time = self.retry_delay * (2 ** (attempts - 1))
                    time.sleep(wait_time)
                else:
                    self._handle_error(e, "Maximale Anzahl von Wiederholungsversuchen erreicht")
                    return None
            except ccxt.ExchangeError as e:
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
        self.error_count += 1
        error_msg = f"{context}: {str(exception)}"
        self.logger.error(error_msg)
        if self.consecutive_errors >= 5:
            self.logger.critical(f"Zu viele aufeinanderfolgende Fehler ({self.consecutive_errors}). Trading wird vorübergehend deaktiviert.")
            self.is_trading_active = False
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
    
    def start_trading(self, mode: Optional[str] = None):
        """
        Aktiviert den Handelsmodus und startet Hintergrund-Tasks.
        Optional kann ein Trading-Modus (z. B. 'live', 'paper') übergeben werden.
        """
        if not self.is_ready:
            self.logger.error("Trading kann nicht gestartet werden - Connector ist nicht bereit.")
            return False
        self.is_trading_active = True
        if mode:
            self.logger.info(f"Trading-Modus: {mode}")
        else:
            self.logger.info("Kein spezieller Trading-Modus angegeben, Standardmodus wird verwendet.")
        self._start_background_tasks()
        self.logger.info("Live Trading aktiviert")
        return True
    
    def stop_trading(self):
        """Deaktiviert den Handelsmodus und stoppt Hintergrund-Tasks."""
        self.is_trading_active = False
        self._stop_background_tasks()
        self.logger.info("Live Trading deaktiviert")
        return True
    
    def _start_background_tasks(self):
        self.background_threads['account_monitor'] = threading.Thread(
            target=self._account_monitor_loop,
            daemon=True
        )
        self.background_threads['orderbook_cache'] = threading.Thread(
            target=self._orderbook_cache_loop,
            daemon=True
        )
        self.background_threads['order_monitor'] = threading.Thread(
            target=self._order_monitor_loop,
            daemon=True
        )
        for thread_name, thread in self.background_threads.items():
            thread.start()
            self.logger.debug(f"Hintergrund-Thread '{thread_name}' gestartet")
    
    def _stop_background_tasks(self):
        self.background_threads = {}
        self.logger.debug("Alle Hintergrund-Threads gestoppt")
    
    def _account_monitor_loop(self):
        while self.is_trading_active:
            try:
                self.get_account_balance()
                self.get_open_positions()
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Fehler im Account-Monitor-Loop: {str(e)}")
                time.sleep(10)
    
    def _orderbook_cache_loop(self):
        common_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT']
        while self.is_trading_active:
            try:
                for symbol in common_symbols:
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
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Fehler im Orderbook-Cache-Loop: {str(e)}")
                time.sleep(5)
    
    def _order_monitor_loop(self):
        while self.is_trading_active:
            try:
                open_orders = self.get_open_orders()
                if open_orders:
                    self.logger.debug(f"Aktuell {len(open_orders)} offene Order(s)")
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Fehler im Order-Monitor-Loop: {str(e)}")
                time.sleep(10)
    
    def get_exchange_info(self):
        return self._execute_with_retry(self.exchange.fetch_markets)
    
    def get_symbol_info(self, symbol: str):
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        if not self.exchange.markets:
            self.exchange.load_markets()
        symbol_info = self.exchange.market(symbol)
        self.symbol_info_cache[symbol] = symbol_info
        return symbol_info
    
    def get_ticker(self, symbol: str):
        return self._execute_with_retry(self.exchange.fetch_ticker, symbol)
    
    def get_orderbook(self, symbol: str, limit: int = 20):
        now = datetime.now()
        if (symbol in self.orderbook_cache and 
            (now - self.orderbook_cache[symbol]['timestamp']).total_seconds() < self.cache_ttl):
            return self.orderbook_cache[symbol]['data']
        orderbook = self._execute_with_retry(self.exchange.fetch_order_book, symbol, limit=limit)
        if orderbook:
            self.orderbook_cache[symbol] = {'data': orderbook, 'timestamp': now}
        return orderbook
    
    def get_account_balance(self):
        balance = self._execute_with_retry(self.exchange.fetch_balance, {'type': 'swap'}, rate_limit_type='orders')
        if balance:
            total_usdt = balance.get('total', {}).get('USDT', 0)
            free_usdt = balance.get('free', {}).get('USDT', 0)
            self.logger.debug(f"Kontostand: {total_usdt} USDT (Verfügbar: {free_usdt} USDT)")
        return balance
    
    def get_open_positions(self):
        positions = self._execute_with_retry(self.exchange.fetch_positions, None, rate_limit_type='orders')
        if positions:
            active_positions = [p for p in positions if float(p['contracts']) > 0]
            if active_positions:
                self.logger.debug(f"Offene Positionen: {len(active_positions)}")
                for pos in active_positions:
                    self.logger.debug(f"  {pos['symbol']}: {pos['side']} {pos['contracts']} Kontrakte, PnL: {pos.get('unrealizedPnl', 'N/A')}")
            return active_positions
        return []
    
    def get_open_orders(self, symbol: Optional[str] = None):
        if symbol:
            return self._execute_with_retry(self.exchange.fetch_open_orders, symbol, rate_limit_type='orders')
        else:
            return self._execute_with_retry(self.exchange.fetch_open_orders, None, rate_limit_type='orders')
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50):
        params = {'limit': limit}
        if symbol:
            return self._execute_with_retry(self.exchange.fetch_orders, symbol, params=params, rate_limit_type='orders')
        else:
            try:
                return self._execute_with_retry(self.exchange.fetch_orders, None, params=params, rate_limit_type='orders')
            except Exception as e:
                self.logger.warning(f"Fehler beim Abrufen der Orderhistorie ohne Symbol: {str(e)}")
                return []
    
    def create_market_order(self, symbol: str, side: str, amount: float, 
                          reduce_only: bool = False, params: Dict = None):
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Market-Order wurde nicht erstellt.")
            return None
        if not params:
            params = {}
        if reduce_only:
            params['reduceOnly'] = True
        order_description = f"{side.upper()} {amount} {symbol} zum Marktpreis"
        self.logger.info(f"Erstelle Market-Order: {order_description}")
        try:
            order = self._execute_with_retry(
                self.exchange.create_market_order,
                symbol, side, amount, None, params,
                rate_limit_type='orders'
            )
            if order:
                self.logger.info(f"Market-Order erfolgreich erstellt: {order['id']}")
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
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Limit-Order wurde nicht erstellt.")
            return None
        if not params:
            params = {}
        if post_only:
            params['postOnly'] = True
        if reduce_only:
            params['reduceOnly'] = True
        order_description = f"{side.upper()} {amount} {symbol} zum Limit-Preis {price}"
        self.logger.info(f"Erstelle Limit-Order: {order_description}")
        try:
            order = self._execute_with_retry(
                self.exchange.create_limit_order,
                symbol, side, amount, price, params,
                rate_limit_type='orders'
            )
            if order:
                self.logger.info(f"Limit-Order erfolgreich erstellt: {order['id']}")
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
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Stop-Loss-Order wurde nicht erstellt.")
            return None
        if not params:
            params = {}
        params['stopPrice'] = stop_price
        params['reduceOnly'] = True
        order_description = f"{side.upper()} {amount} {symbol} bei Stop-Loss {stop_price}"
        self.logger.info(f"Erstelle Stop-Loss-Order: {order_description}")
        try:
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'stop_market', side, amount, None, params,
                rate_limit_type='orders'
            )
            if order:
                self.logger.info(f"Stop-Loss-Order erfolgreich erstellt: {order['id']}")
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
        if not self.is_trading_active:
            self.logger.warning("Trading ist deaktiviert. Take-Profit-Order wurde nicht erstellt.")
            return None
        if not params:
            params = {}
        params['stopPrice'] = take_profit_price
        params['reduceOnly'] = True
        order_description = f"{side.upper()} {amount} {symbol} bei Take-Profit {take_profit_price}"
        self.logger.info(f"Erstelle Take-Profit-Order: {order_description}")
        try:
            order = self._execute_with_retry(
                self.exchange.create_order,
                symbol, 'take_profit_market', side, amount, None, params,
                rate_limit_type='orders'
            )
            if order:
                self.logger.info(f"Take-Profit-Order erfolgreich erstellt: {order['id']}")
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
        try:
            self.logger.info(f"Storniere Order {order_id} für {symbol}")
            result = self._execute_with_retry(
                self.exchange.cancel_order,
                order_id, symbol,
                rate_limit_type='orders'
            )
            if result:
                self.logger.info(f"Order {order_id} erfolgreich storniert")
                for callback in self.order_update_callbacks:
                    try:
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
                open_orders = self.get_open_orders()
                if not open_orders:
                    return {'canceled': 0}
                orders_by_symbol = {}
                for order in open_orders:
                    sym = order['symbol']
                    orders_by_symbol.setdefault(sym, []).append(order)
                total_canceled = 0
                for sym, orders in orders_by_symbol.items():
                    canceled = self._execute_with_retry(
                        self.exchange.cancel_all_orders,
                        sym,
                        rate_limit_type='orders'
                    )
                    if canceled and isinstance(canceled, dict) and 'canceled' in canceled:
                        total_canceled += canceled['canceled']
                    else:
                        total_canceled += len(orders)
                result = {'canceled': total_canceled}
            return result
        except Exception as e:
            error_msg = f"Fehler beim Stornieren aller Orders: {str(e)}"
            self._handle_error(e, error_msg)
            return None
    
    def register_error_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.error_callbacks.append(callback)
    
    def register_order_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.order_update_callbacks.append(callback)
    
    def register_position_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.position_update_callbacks.append(callback)
    
    def close_all_positions(self):
        try:
            self.logger.info("Schließe alle offenen Positionen...")
            open_positions = self.get_open_positions()
            results = []
            for pos in open_positions:
                symbol = pos['symbol']
                side = 'sell' if pos['side'].lower() == 'long' else 'buy'
                amount = pos['contracts']
                result = self.create_market_order(symbol, side, amount, reduce_only=True)
                results.append(result)
            return results
        except Exception as e:
            error_msg = f"Fehler beim Schließen aller Positionen: {str(e)}"
            self._handle_error(e, error_msg)
            return None

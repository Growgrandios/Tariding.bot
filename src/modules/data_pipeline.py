import os
import logging
import time
import threading
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import ccxt
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from dotenv import load_dotenv
import websocket
from bs4 import BeautifulSoup
import xmltodict

class DataPipeline:
    """
    Zentrale Komponente zur Beschaffung, Verarbeitung und Speicherung von Marktdaten
    aus verschiedenen Quellen (Krypto-Börsen, Aktien-APIs, etc.).
    """

    def __init__(self, api_keys: Dict[str, Any] = None):
        """
        Initialisiert die Datenpipeline.
        Args:
            api_keys: Dictionary mit API-Schlüsseln für verschiedene Datenquellen.
        """
        self.logger = logging.getLogger("DataPipeline")
        self.logger.info("Initialisiere DataPipeline...")

        # Lade API-Schlüssel aus .env, falls nicht explizit übergeben.
        load_dotenv()
        self.api_keys = api_keys or {}

        # Setze Default-API-Keys aus Umgebungsvariablen, falls nicht übergeben.
        if 'alpha_vantage' not in self.api_keys:
            self.api_keys['alpha_vantage'] = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if 'bitget' not in self.api_keys:
            self.api_keys['bitget'] = {
                'api_key': os.getenv("BITGET_API_KEY", ""),
                'api_secret': os.getenv("BITGET_API_SECRET", ""),
                'api_passphrase': os.getenv("BITGET_API_PASSPHRASE", "")
            }

        # Weitere API-Schlüssel für zusätzliche Datenquellen.
        if 'prorealtime' not in self.api_keys:
            self.api_keys['prorealtime'] = {
                'username': os.getenv("PROREALTIME_USERNAME", ""),
                'password': os.getenv("PROREALTIME_PASSWORD", "")
            }
        if 'polygon' not in self.api_keys:
            self.api_keys['polygon'] = os.getenv("POLYGON_API_KEY", "")
        if 'news_api' not in self.api_keys:
            self.api_keys['news_api'] = os.getenv("NEWS_API_KEY", "")

        # Initialisiere Datenquellen.
        self._init_data_sources()

        # Zwischenspeicher für Daten (Cache).
        self.data_cache = {
            'crypto': {},
            'stocks': {},
            'forex': {},
            'macro': {},
            'news': {}
        }

        # Timer für automatische Datenaktualisierung.
        self.update_threads = {}
        self.update_intervals = {
            'crypto': 60,  # Sekunden.
            'stocks': 300,  # Sekunden.
            'forex': 300,  # Sekunden.
            'macro': 86400,  # Sekunden (täglich).
            'news': 3600  # Sekunden (stündlich).
        }

        # Status-Flags.
        self.is_running = False
        self.last_update = {
            'crypto': None,
            'stocks': None,
            'forex': None,
            'macro': None,
            'news': None
        }

        self.logger.info("DataPipeline erfolgreich initialisiert.")

    def _init_data_sources(self):
        """
        Initialisiert Verbindungen zu verschiedenen Datenquellen.
        """
        self.sources = {}

        # CCXT für Krypto-Börsen.
        try:
            # Bitget für Kryptowährungen.
            if all(self.api_keys['bitget'].values()):
                self.sources['bitget'] = ccxt.bitget({
                    'apiKey': self.api_keys['bitget']['api_key'],
                    'secret': self.api_keys['bitget']['api_secret'],
                    'password': self.api_keys['bitget']['api_passphrase'],
                    'enableRateLimit': True
                })
                self.logger.info("Bitget API erfolgreich initialisiert.")
            else:
                self.sources['bitget'] = ccxt.bitget({'enableRateLimit': True})
                self.logger.warning("Bitget API im öffentlichen Modus initialisiert (keine API-Schlüssel).")

            # Weitere Krypto-Börsen für Preisvergleiche.
            self.sources['binance'] = ccxt.binance({'enableRateLimit': True})
            self.sources['coinbase'] = ccxt.coinbase({'enableRateLimit': True})
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Krypto-Börsen: {str(e)}")

        # Alpha Vantage für Aktien, Forex und Makrodaten.
        try:
            if self.api_keys['alpha_vantage']:
                self.sources['alpha_vantage_stocks'] = TimeSeries(
                    key=self.api_keys['alpha_vantage'],
                    output_format='pandas'
                )
                self.sources['alpha_vantage_crypto'] = CryptoCurrencies(
                    key=self.api_keys['alpha_vantage'],
                    output_format='pandas'
                )
                self.logger.info("Alpha Vantage API erfolgreich initialisiert.")
            else:
                self.logger.warning("Alpha Vantage API nicht initialisiert (kein API-Schlüssel).")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung von Alpha Vantage: {str(e)}")

        # Yahoo Finance (kein API-Key erforderlich).
        self.sources['yfinance'] = True

        # Polygon.io für erweiterte Marktdaten.
        try:
            if self.api_keys.get('polygon'):
                self.sources['polygon'] = {
                    'api_key': self.api_keys['polygon'],
                    'base_url': 'https://api.polygon.io/v2'
                }
                self.logger.info("Polygon.io API erfolgreich initialisiert.")
            else:
                self.logger.warning("Polygon.io API nicht initialisiert (kein API-Schlüssel).")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung von Polygon.io: {str(e)}")

        # ProRealTime für Backtesting und Paper-Trading.
        try:
            if all(self.api_keys.get('prorealtime', {}).values()):
                try:
                    from src.modules.prorealtime_connector import ProRealTimeConnector  # Import hier benötigt.
                    self.sources['prorealtime'] = ProRealTimeConnector(
                        credentials=self.api_keys['prorealtime']
                    )
                    self.logger.info("ProRealTime API erfolgreich initialisiert.")
                except ImportError:
                    self.logger.warning("ProRealTime-Modul nicht gefunden. ProRealTime-Features werden deaktiviert.")
            else:
                self.logger.warning("ProRealTime API nicht initialisiert (keine Zugangsdaten).")
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung von ProRealTime: {str(e)}")

    def start_auto_updates(self):
        """Startet automatische Datenaktualisierungen in separaten Threads."""
        if self.is_running:
            self.logger.warning("Automatische Updates laufen bereits")
            return

        self.is_running = True
        for data_type, interval in self.update_intervals.items():
            self.update_threads[data_type] = threading.Thread(
                target=self._auto_update_loop,
                args=(data_type, interval),
                daemon=True
            )
            self.update_threads[data_type].start()
            self.logger.info(f"Auto-Update für {data_type} gestartet (Intervall: {interval}s)")

    def stop_auto_updates(self):
        """Stoppt alle automatischen Datenaktualisierungen."""
        self.is_running = False
        # Warten, bis alle Threads beendet sind
        for data_type, thread in self.update_threads.items():
            if thread.is_alive():
                thread.join(timeout=10)
        self.logger.info("Alle Auto-Updates gestoppt")

    def _auto_update_loop(self, data_type: str, interval: int):
        """
        Aktualisierungsschleife für einen bestimmten Datentyp.
        Args:
            data_type: Art der Daten ('crypto', 'stocks', etc.)
            interval: Aktualisierungsintervall in Sekunden
        """
        while self.is_running:
            try:
                self._update_data(data_type)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Fehler im Update-Loop für {data_type}: {str(e)}")
                time.sleep(interval)  # Trotz Fehler reguläres Intervall einhalten

    def _update_data(self, data_type: str):
        """
        Aktualisiert Daten eines bestimmten Typs.
        Args:
            data_type: Art der Daten ('crypto', 'stocks', etc.)
        """
        if data_type == 'crypto':
            self._update_crypto_data()
        elif data_type == 'stocks':
            self._update_stock_data()
        elif data_type == 'forex':
            self._update_forex_data()
        elif data_type == 'macro':
            self._update_macro_data()
        elif data_type == 'news':
            self._update_news_data()
        else:
            self.logger.warning(f"Unbekannter Datentyp: {data_type}")

        self.last_update[data_type] = datetime.now()

    def _update_crypto_data(self):
        """Aktualisiert Kryptowährungsdaten."""
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT'
        ]
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

        for symbol in symbols:
            symbol_data = {}
            for timeframe in timeframes:
                try:
                    # Primäre Quelle: Bitget
                    ohlcv = self.sources['bitget'].fetch_ohlcv(symbol, timeframe)
                    if ohlcv and len(ohlcv) > 0:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        # Als Zeitindex setzen
                        df.set_index('timestamp', inplace=True)
                        # Im Cache speichern
                        symbol_data[timeframe] = df
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen von OHLCV-Daten für {symbol} ({timeframe}): {str(e)}")
                    # Fallback: Versuche alternative Quelle
                    try:
                        if 'binance' in self.sources:
                            ohlcv = self.sources['binance'].fetch_ohlcv(symbol, timeframe)
                            if ohlcv and len(ohlcv) > 0:
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df.set_index('timestamp', inplace=True)
                                # Im Cache speichern
                                symbol_data[timeframe] = df
                                self.logger.info(f"Fallback-Daten für {symbol} ({timeframe}) von Binance verwendet")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback fehlgeschlagen für {symbol}: {str(fallback_error)}")

            if symbol_data:
                self.data_cache['crypto'][symbol] = symbol_data
                self.logger.debug(f"Kryptodaten für {symbol} aktualisiert")

        self.logger.info(f"Kryptodaten für {len(symbols)} Symbole aktualisiert")

    def _update_stock_data(self):
        """Aktualisiert Aktiendaten."""
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'
        ]
        intervals = ['1d', '1wk', '1mo']

        for symbol in symbols:
            symbol_data = {}
            try:
                # Yahoo Finance für Aktiendaten
                stock = yf.Ticker(symbol)
                for interval in intervals:
                    # Historische Daten abrufen
                    hist = stock.history(period='1y', interval=interval)
                    if not hist.empty:
                        # Im Cache speichern
                        symbol_data[interval] = hist
                        self.logger.debug(f"Aktiendaten für {symbol} ({interval}) aktualisiert")

                # Fundamentaldaten hinzufügen, falls verfügbar
                try:
                    info = stock.info
                    symbol_data['info'] = info
                except:
                    pass

                if symbol_data:
                    self.data_cache['stocks'][symbol] = symbol_data
            except Exception as e:
                self.logger.error(f"Fehler beim Abrufen von Aktiendaten für {symbol}: {str(e)}")
                # Fallback zu Alpha Vantage, falls verfügbar
                if 'alpha_vantage_stocks' in self.sources:
                    try:
                        data, meta_data = self.sources['alpha_vantage_stocks'].get_daily(symbol=symbol, outputsize='full')
                        symbol_data['1d'] = data
                        self.data_cache['stocks'][symbol] = symbol_data
                        self.logger.info(f"Fallback-Daten für {symbol} von Alpha Vantage verwendet")
                    except Exception as fallback_error:
                        self.logger.error(f"Alpha Vantage Fallback fehlgeschlagen: {str(fallback_error)}")

        self.logger.info(f"Aktiendaten für {len(symbols)} Symbole aktualisiert")

    def _update_forex_data(self):
        """Aktualisiert Forex-Daten."""
        pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF']

        for pair in pairs:
            try:
                symbol = f"{pair}=X"  # Yahoo Finance Format
                forex = yf.Ticker(symbol)
                hist = forex.history(period='1y')
                if not hist.empty:
                    self.data_cache['forex'][pair] = hist
                    self.logger.debug(f"Forex-Daten für {pair} aktualisiert")
            except Exception as e:
                self.logger.error(f"Fehler beim Abrufen von Forex-Daten für {pair}: {str(e)}")

        self.logger.info(f"Forex-Daten für {len(pairs)} Paare aktualisiert")

    def _update_macro_data(self):
        """Aktualisiert makroökonomische Daten."""
        # Bei realer Implementierung würden hier Daten von APIs wie FRED,
        # Alpha Vantage, oder spezialisierten Wirtschaftsdaten-APIs kommen
        # Dummy-Implementierung für Testzwecke
        macro_indicators = {
            'inflation_rate': 3.1,
            'unemployment_rate': 3.7,
            'interest_rate': 5.25,
            'gdp_growth': 2.1,
            'consumer_sentiment': 67.3,
            'leading_index': -0.2,
            'yield_curve': {
                '3m': 5.48,
                '2y': 5.11,
                '5y': 4.77,
                '10y': 4.83,
                '30y': 4.95
            }
        }
        self.data_cache['macro'] = macro_indicators
        self.logger.info("Makroökonomische Daten aktualisiert")

    def _update_news_data(self):
        """Aktualisiert Finanznachrichten."""
        # Implementieren wir die Live-Nachrichten-API hier
        try:
            if 'news_api' in self.api_keys and self.api_keys['news_api']:
                news = self.get_live_market_news(
                    tickers=['BTC', 'ETH', 'CRYPTO', 'FOREX', 'STOCK', 'MARKET'],
                    limit=20
                )
                if news:
                    self.logger.info(f"{len(news)} aktuelle Nachrichten abgerufen")
            else:
                self.logger.warning("News API Key nicht konfiguriert, keine Nachrichtenaktualisierung möglich")
        except Exception as e:
            self.logger.error(f"Fehler bei der Nachrichtenaktualisierung: {str(e)}")
            # Dummy-Daten als Fallback
        self.data_cache['news'] = []
        self.logger.info("Nachrichtendaten aktualisiert")

    # Öffentliche Methoden für den Datenzugriff

    def get_crypto_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Ruft aktuelle Kryptowährungsdaten ab.
        Args:
            symbol: Trading-Symbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Anzahl der Datenpunkte
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Prüfen, ob Daten im Cache vorhanden sind
            if (symbol in self.data_cache['crypto'] and
                    timeframe in self.data_cache['crypto'][symbol]):
                data = self.data_cache['crypto'][symbol][timeframe]
                return data.tail(limit)

            # Falls nicht im Cache, direkt von der Börse abrufen
            ohlcv = self.sources['bitget'].fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Kryptodaten für {symbol}: {str(e)}")
            return None

    def get_stock_data(self, symbol: str, interval: str = '1d', period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Ruft aktuelle Aktiendaten ab.
        Args:
            symbol: Aktien-Symbol (z.B. 'AAPL')
            interval: Zeitintervall ('1d', '1wk', '1mo')
            period: Zeitraum ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Prüfen, ob Daten im Cache vorhanden sind
            if (symbol in self.data_cache['stocks'] and
                    interval in self.data_cache['stocks'][symbol]):
                return self.data_cache['stocks'][symbol][interval]

            # Falls nicht im Cache, direkt abrufen
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            if not hist.empty:
                return hist
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Aktiendaten für {symbol}: {str(e)}")
            return None

    def fetch_crypto_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Ruft historische Kryptowährungsdaten für einen bestimmten Zeitraum ab.
        Args:
            symbol: Trading-Symbol (z.B. 'BTC/USDT')
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'
            interval: Zeitintervall ('1h', '4h', '1d')
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Konvertiere Datumsangaben
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)

            # Bestimme die ungefähre Anzahl der benötigten Datenpunkte
            if interval == '1h':
                days = (end_datetime - start_datetime).days
                limit = days * 24  # 24 Stunden pro Tag
            elif interval == '4h':
                days = (end_datetime - start_datetime).days
                limit = days * 6  # 6 4-Stunden-Intervalle pro Tag
            else:  # '1d'
                limit = (end_datetime - start_datetime).days

            # Setze ein vernünftiges Maximum
            limit = min(limit, 1000)

            # Versuche zuerst, aus dem Cache zu laden
            if (symbol in self.data_cache['crypto'] and
                    interval in self.data_cache['crypto'][symbol]):
                df = self.data_cache['crypto'][symbol][interval]
                # Filtere den gewünschten Zeitraum
                filtered_df = df.loc[start_datetime:end_datetime]
                if not filtered_df.empty:
                    return filtered_df

            # Direkt von der Börse abrufen
            since = int(start_datetime.timestamp() * 1000)  # in Millisekunden
            ohlcv = self.sources['bitget'].fetch_ohlcv(symbol, interval, since=since, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                # Filtere den gewünschten Zeitraum
                return df.loc[start_datetime:end_datetime]
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen historischer Kryptodaten: {str(e)}")
            return None

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Ruft historische Aktiendaten für einen bestimmten Zeitraum ab.
        Args:
            symbol: Aktien-Symbol (z.B. 'AAPL')
            start_date: Startdatum im Format 'YYYY-MM-DD'
            end_date: Enddatum im Format 'YYYY-MM-DD'
            interval: Zeitintervall ('1d', '1wk', '1mo')
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Versuche, aus dem Cache zu laden
            if (symbol in self.data_cache['stocks'] and
                    interval in self.data_cache['stocks'][symbol]):
                df = self.data_cache['stocks'][symbol][interval]
                # Filtere den gewünschten Zeitraum
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(end_date)
                filtered_df = df.loc[start_datetime:end_datetime]
                if not filtered_df.empty:
                    return filtered_df

            # Abrufen von Yahoo Finance
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date, interval=interval)
            if not hist.empty:
                return hist
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen historischer Aktiendaten: {str(e)}")
            return None

    def get_forex_data(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Ruft aktuelle Forex-Daten ab.
        Args:
            pair: Währungspaar (z.B. 'EURUSD')
        Returns:
            DataFrame mit Kursdaten oder None bei Fehler
        """
        try:
            if pair in self.data_cache['forex']:
                return self.data_cache['forex'][pair]

            # Direkt abrufen, falls nicht im Cache
            symbol = f"{pair}=X"  # Yahoo Finance Format
            forex = yf.Ticker(symbol)
            hist = forex.history(period='1y')
            if not hist.empty:
                return hist
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Forex-Daten für {pair}: {str(e)}")
            return None

    def get_macro_indicators(self) -> Dict[str, Any]:
        """
        Ruft makroökonomische Indikatoren ab.
        Returns:
            Dictionary mit Indikatoren
        """
        return self.data_cache['macro']

    def get_market_sentiment(self) -> Dict[str, float]:
        """
        Berechnet und gibt Marktstimmungsindikatoren zurück.
        Returns:
            Dictionary mit Stimmungsindikatoren
        """
        sentiment = {
            'vix': 20.5,  # Volatilitätsindex (simuliert)
            'put_call_ratio': 0.85,
            'bull_bear_ratio': 1.2,
            'crypto_fear_greed': 55,  # 0-100 Skala
            'average_sentiment': 0.3  # -1 bis 1 Skala
        }
        return sentiment

    def get_liquidity_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Ruft Liquiditätsmetriken für verschiedene Assets ab.
        Returns:
            Dictionary mit Liquiditätsmetriken
        """
        # Implementierung für reale Anwendung würde Daten aus Orderbooks abrufen
        # Hier Dummy-Daten
        metrics = {
            'BTC/USDT': {
                'spread': 0.15,
                'avg_spread': 0.2,
                'volume': 1500000000,
                'avg_volume': 1200000000,
                'market_depth': 25000000,
                'avg_market_depth': 20000000
            },
            'ETH/USDT': {
                'spread': 0.12,
                'avg_spread': 0.15,
                'volume': 800000000,
                'avg_volume': 900000000,
                'market_depth': 15000000,
                'avg_market_depth': 18000000
            }
        }
        return metrics

    def get_indicators(self, data: pd.DataFrame, indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Berechnet technische Indikatoren für einen Datensatz.
        Args:
            data: DataFrame mit OHLCV-Daten
            indicators: Liste von Indikatoren mit Parametern
        Returns:
            DataFrame mit hinzugefügten Indikatoren
        """
        # Kopie des DataFrames erstellen
        df = data.copy()

        for indicator in indicators:
            try:
                indicator_type = indicator['type']
                if indicator_type == 'sma':
                    # Simple Moving Average
                    period = indicator.get('period', 20)
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                elif indicator_type == 'ema':
                    # Exponential Moving Average
                    period = indicator.get('period', 20)
                    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                elif indicator_type == 'rsi':
                    # Relative Strength Index
                    period = indicator.get('period', 14)
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    rs = avg_gain / avg_loss
                    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                elif indicator_type == 'bollinger':
                    # Bollinger Bands
                    period = indicator.get('period', 20)
                    std_dev = indicator.get('std_dev', 2)
                    df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'bb_std_{period}'] = df['close'].rolling(window=period).std()
                    df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + std_dev * df[f'bb_std_{period}']
                    df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - std_dev * df[f'bb_std_{period}']
                elif indicator_type == 'macd':
                    # MACD
                    fast_period = indicator.get('fast_period', 12)
                    slow_period = indicator.get('slow_period', 26)
                    signal_period = indicator.get('signal_period', 9)
                    df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period, adjust=False).mean()
                    df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period, adjust=False).mean()
                    df['macd'] = df[f'ema_{fast_period}'] - df[f'ema_{slow_period}']
                    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                elif indicator_type == 'atr':
                    # Average True Range
                    period = indicator.get('period', 14)
                    high_low = df['high'] - df['low']
                    high_close = (df['high'] - df['close'].shift()).abs()
                    low_close = (df['low'] - df['close'].shift()).abs()
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    df[f'atr_{period}'] = true_range.rolling(window=period).mean()
                else:
                    self.logger.warning(f"Unbekannter Indikator-Typ: {indicator_type}")
            except Exception as e:
                self.logger.error(f"Fehler bei der Berechnung des Indikators {indicator['type']}: {str(e)}")

        return df

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Ruft das aktuelle Orderbuch für ein Symbol ab.
        Args:
            symbol: Trading-Symbol (z.B. 'BTC/USDT')
            limit: Tiefe des Orderbuchs
        Returns:
            Dictionary mit Orderbuch-Daten
        """
        try:
            order_book = self.sources['bitget'].fetch_order_book(symbol, limit)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': datetime.now().isoformat(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol
            }
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Orderbuchs für {symbol}: {str(e)}")
            return {
                'bids': [],
                'asks': [],
                'timestamp': datetime.now().isoformat(),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'error': str(e)
            }

    def get_account_balance(self) -> Dict[str, float]:
        """
        Ruft den Kontostand von Bitget ab.
        Returns:
            Dictionary mit Kontoguthaben
        """
        try:
            if all(self.api_keys['bitget'].values()):
                balances = self.sources['bitget'].fetch_balance()
                # Extrahiere die relevanten Informationen
                account_balance = {}
                for currency, data in balances['total'].items():
                    if isinstance(data, (int, float)) and data > 0:
                        account_balance[currency] = data
                return account_balance
            else:
                self.logger.warning("Keine API-Schlüssel für Bitget vorhanden, Kontostand kann nicht abgerufen werden")
                return {}
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des Kontostands: {str(e)}")
            return {}

    def get_last_update_time(self, data_type: str) -> Optional[datetime]:
        """
        Gibt den Zeitpunkt der letzten Aktualisierung für einen Datentyp zurück.
        Args:
            data_type: Art der Daten ('crypto', 'stocks', etc.)
        Returns:
            Zeitpunkt der letzten Aktualisierung oder None
        """
        return self.last_update.get(data_type)

    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den Status der Datenpipeline zurück.
        Returns:
            Status-Dictionary
        """
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'cache_size': {
                'crypto': len(self.data_cache['crypto']),
                'stocks': len(self.data_cache['stocks']),
                'forex': len(self.data_cache['forex']),
                'macro': bool(self.data_cache['macro']),
                'news': len(self.data_cache['news'])
            }
        }

    # Neue Methoden für Backtesting, Paper-Trading und Echtzeit-Daten

    def run_backtest(self, strategy_id, parameters=None, start_date=None, end_date=None, symbols=None):
        """
        Führt einen Backtest mit ProRealTime aus.
        Args:
            strategy_id: ID der zu testenden Strategie
            parameters: Parameter für die Strategie
            start_date: Startdatum für den Backtest
            end_date: Enddatum für den Backtest
            symbols: Liste von Symbolen für den Backtest
        Returns:
            Dictionary mit Backtest-Ergebnissen oder None bei Fehler
        """
        self.logger.info(f"Starte Backtest für Strategie {strategy_id}")
        if 'prorealtime' in self.sources:
            try:
                result = self.sources['prorealtime'].run_backtest(
                    strategy_id=strategy_id,
                    parameters=parameters,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols
                )
                return result
            except Exception as e:
                self.logger.error(f"Fehler beim Ausführen des Backtests: {str(e)}")
                return None
        else:
            self.logger.error("ProRealTime nicht initialisiert, Backtest nicht möglich")
            return None

    def start_paper_trading(self, strategy_id, parameters=None):
        """
        Startet Paper-Trading mit einer bestimmten Strategie.
        Args:
            strategy_id: ID der zu verwendenden Strategie
            parameters: Parameter für die Strategie
        Returns:
            True bei Erfolg, False bei Fehler
        """
        self.logger.info(f"Starte Paper-Trading für Strategie {strategy_id}")
        if 'prorealtime' in self.sources:
            try:
                result = self.sources['prorealtime'].start_paper_trading(
                    strategy_id=strategy_id,
                    parameters=parameters
                )
                return result
            except Exception as e:
                self.logger.error(f"Fehler beim Starten des Paper-Trading: {str(e)}")
                return False
        else:
            self.logger.error("ProRealTime nicht initialisiert, Paper-Trading nicht möglich")
            return False

    def get_realtime_forex_data(self, symbol):
        """
        Ruft Echtzeit-Forex-Daten von Polygon.io ab.
        Args:
            symbol: Währungspaar (z.B. 'EUR/USD')
        Returns:
            Dictionary mit Echtzeit-Daten oder None bei Fehler
        """
        if 'polygon' not in self.sources:
            self.logger.error("Polygon.io nicht initialisiert")
            return None

        try:
            # Formate Symbol für Polygon.io (EUR/USD -> C:EURUSD)
            formatted_symbol = f"C:{symbol.replace('/', '')}"
            url = f"{self.sources['polygon']['base_url']}/last/forex/{formatted_symbol}"
            params = {
                'apiKey': self.sources['polygon']['api_key']
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                self.logger.error(f"Fehler beim Abrufen von Echtzeit-Forex-Daten: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Echtzeit-Forex-Daten: {str(e)}")
            return None

    def get_live_market_news(self, tickers=None, limit=10):
        """
        Ruft aktuelle Marktnachrichten für bestimmte Tickers ab.
        Args:
            tickers: Liste von Tickersymbolen
            limit: Maximale Anzahl von Nachrichten
        Returns:
            Liste mit Nachrichten oder leere Liste bei Fehler
        """
        if 'news_api' not in self.api_keys or not self.api_keys['news_api']:
            self.logger.error("News API Key nicht konfiguriert")
            return []

        try:
            url = "https://newsapi.org/v2/everything"
            # Erstelle Abfrage basierend auf den Tickers
            query = " OR ".join([f'"{ticker}"' for ticker in tickers]) if tickers else "finance OR stock OR crypto"
            params = {
                'q': query,
                'language': 'de,en',
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'apiKey': self.api_keys['news_api']
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                # Speichere im Cache
                self.data_cache['news'] = data.get('articles', [])
                self.last_update['news'] = datetime.now()
                return data.get('articles', [])
            else:
                self.logger.error(f"Fehler beim Abrufen von Marktnachrichten: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Marktnachrichten: {str(e)}")
            return []

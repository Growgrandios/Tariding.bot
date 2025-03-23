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
            'crypto': 60,     # Sekunden.
            'stocks': 300,    # Sekunden.
            'forex': 300,     # Sekunden.
            'macro': 86400,   # Sekunden (täglich).
            'news': 3600      # Sekunden (stündlich).
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
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'LTC/USDT'
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
                        self.logger.error(f"Alpha Vantage Fallback fehlgeschlagen für {symbol}: {str(fallback_error)}")

    def _update_forex_data(self):
        """Aktualisiert Forex-Daten."""
        self.logger.info("Aktualisiere Forex-Daten...")
        try:
            # Forex-Paare definieren
            forex_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']
            
            for pair in forex_pairs:
                try:
                    # Alpha Vantage für Forex-Daten verwenden (falls verfügbar)
                    if 'alpha_vantage' in self.api_keys and self.api_keys['alpha_vantage']:
                        from_currency, to_currency = pair.split('/')
                        # API-Anfrage für Forex-Daten
                        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={self.api_keys['alpha_vantage']}"
                        response = requests.get(url)
                        data = response.json()
                        
                        if 'Time Series FX (Daily)' in data:
                            # Daten in DataFrame konvertieren
                            time_series = data['Time Series FX (Daily)']
                            df = pd.DataFrame.from_dict(time_series, orient='index')
                            df.index = pd.to_datetime(df.index)
                            df = df.astype(float)
                            # Spalten umbenennen
                            df.columns = ['open', 'high', 'low', 'close']
                            # Im Cache speichern
                            self.data_cache['forex'][pair] = {'1d': df}
                            self.logger.debug(f"Forex-Daten für {pair} aktualisiert")
                    else:
                        self.logger.warning(f"Alpha Vantage API-Schlüssel fehlt - kann keine Forex-Daten für {pair} abrufen")
                
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen von Forex-Daten für {pair}: {str(e)}")
            
            self.logger.info("Forex-Daten aktualisiert")
        
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der Forex-Daten: {str(e)}")

    def _update_macro_data(self):
        """Aktualisiert makroökonomische Daten."""
        self.logger.info("Aktualisiere makroökonomische Daten...")
        try:
            # Liste der makroökonomischen Indikatoren
            indicators = ['GDP', 'INFLATION', 'UNEMPLOYMENT', 'INTEREST_RATE']
            
            for indicator in indicators:
                try:
                    # Alpha Vantage für makroökonomische Daten verwenden
                    if 'alpha_vantage' in self.api_keys and self.api_keys['alpha_vantage']:
                        # Abbildung der Indikatoren auf Alpha Vantage-Funktionen
                        function_map = {
                            'GDP': 'REAL_GDP',
                            'INFLATION': 'INFLATION',
                            'UNEMPLOYMENT': 'UNEMPLOYMENT',
                            'INTEREST_RATE': 'FEDERAL_FUNDS_RATE'
                        }
                        
                        function = function_map.get(indicator)
                        if function:
                            # API-Anfrage für makroökonomische Daten
                            url = f"https://www.alphavantage.co/query?function={function}&interval=quarterly&apikey={self.api_keys['alpha_vantage']}"
                            response = requests.get(url)
                            data = response.json()
                            
                            if 'data' in data:
                                # Daten in DataFrame konvertieren
                                df = pd.DataFrame(data['data'])
                                df['date'] = pd.to_datetime(df['date'])
                                df.set_index('date', inplace=True)
                                df['value'] = df['value'].astype(float)
                                # Im Cache speichern
                                self.data_cache['macro'][indicator] = df
                                self.logger.debug(f"Makroökonomische Daten für {indicator} aktualisiert")
                    else:
                        self.logger.warning("Alpha Vantage API-Schlüssel fehlt - kann keine makroökonomischen Daten abrufen")
                
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen von makroökonomischen Daten für {indicator}: {str(e)}")
            
            self.logger.info("Makroökonomische Daten aktualisiert")
        
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der makroökonomischen Daten: {str(e)}")

    def _update_news_data(self):
        """Aktualisiert Nachrichtendaten."""
        self.logger.info("Aktualisiere Nachrichtendaten...")
        try:
            # Kategorien für Nachrichten
            categories = ['crypto', 'stock', 'economy']
            
            for category in categories:
                try:
                    # News API für Nachrichtendaten verwenden
                    if 'news_api' in self.api_keys and self.api_keys['news_api']:
                        # Suchbegriffe je nach Kategorie
                        search_terms = {
                            'crypto': 'cryptocurrency OR bitcoin OR ethereum',
                            'stock': 'stock market OR wall street',
                            'economy': 'economy OR financial OR market'
                        }
                        
                        # API-Anfrage für Nachrichtendaten
                        url = f"https://newsapi.org/v2/everything?q={search_terms[category]}&apiKey={self.api_keys['news_api']}&language=en&pageSize=20"
                        response = requests.get(url)
                        data = response.json()
                        
                        if 'articles' in data:
                            # Nur relevante Felder behalten
                            articles = []
                            for article in data['articles']:
                                articles.append({
                                    'title': article['title'],
                                    'description': article['description'],
                                    'url': article['url'],
                                    'publishedAt': article['publishedAt'],
                                    'source': article['source']['name']
                                })
                            
                            # Im Cache speichern
                            self.data_cache['news'][category] = articles
                            self.logger.debug(f"Nachrichtendaten für {category} aktualisiert")
                    else:
                        self.logger.warning("News API-Schlüssel fehlt - kann keine Nachrichtendaten abrufen")
                
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen von Nachrichtendaten für {category}: {str(e)}")
            
            self.logger.info("Nachrichtendaten aktualisiert")
        
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der Nachrichtendaten: {str(e)}")

    def get_crypto_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Ruft Kryptowährungsdaten für ein bestimmtes Symbol und einen Zeitrahmen ab.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitintervall ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Anzahl der zurückzugebenden Datenpunkte
        
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Prüfen, ob Daten im Cache vorhanden sind
            if symbol in self.data_cache['crypto'] and timeframe in self.data_cache['crypto'][symbol]:
                data = self.data_cache['crypto'][symbol][timeframe]
                # Nur die angeforderte Anzahl zurückgeben
                return data.tail(limit)
            
            # Falls nicht im Cache, direkt von der Börse abrufen
            if 'bitget' in self.sources:
                ohlcv = self.sources['bitget'].fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Im Cache speichern für zukünftige Anfragen
                    if symbol not in self.data_cache['crypto']:
                        self.data_cache['crypto'][symbol] = {}
                    
                    self.data_cache['crypto'][symbol][timeframe] = df
                    return df
            
            # Fallback zu anderen Börsen
            for exchange_name in ['binance', 'coinbase']:
                if exchange_name in self.sources:
                    try:
                        ohlcv = self.sources[exchange_name].fetch_ohlcv(symbol, timeframe, limit=limit)
                        
                        if ohlcv and len(ohlcv) > 0:
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            
                            # Im Cache speichern
                            if symbol not in self.data_cache['crypto']:
                                self.data_cache['crypto'][symbol] = {}
                            
                            self.data_cache['crypto'][symbol][timeframe] = df
                            return df
                    
                    except Exception:
                        continue
            
            self.logger.warning(f"Keine Daten für {symbol} ({timeframe}) gefunden")
            return None
        
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Crypto-Daten für {symbol} ({timeframe}): {str(e)}")
            return None

    def get_stock_data(self, symbol: str, interval: str = '1d', limit: int = 100) -> pd.DataFrame:
        """
        Ruft Aktiendaten für ein bestimmtes Symbol und einen Zeitrahmen ab.
        
        Args:
            symbol: Aktien-Symbol (z.B. 'AAPL')
            interval: Zeitintervall ('1d', '1wk', '1mo')
            limit: Anzahl der zurückzugebenden Datenpunkte
        
        Returns:
            DataFrame mit OHLCV-Daten oder None bei Fehler
        """
        try:
            # Prüfen, ob Daten im Cache vorhanden sind
            if symbol in self.data_cache['stocks'] and interval in self.data_cache['stocks'][symbol]:
                data = self.data_cache['stocks'][symbol][interval]
                # Nur die angeforderte Anzahl zurückgeben
                return data.tail(limit)
            
            # Zeitraum basierend auf Intervall und Limit berechnen
            if interval == '1d':
                period = f"{limit}d"
            elif interval == '1wk':
                period = f"{limit * 7}d"
            elif interval == '1mo':
                period = f"{limit * 30}d"
            else:
                period = '1y'
            
            # Daten über yfinance abrufen
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval=interval)
            
            if not hist.empty:
                # Im Cache speichern für zukünftige Anfragen
                if symbol not in self.data_cache['stocks']:
                    self.data_cache['stocks'][symbol] = {}
                
                self.data_cache['stocks'][symbol][interval] = hist
                return hist
            
            # Fallback zu Alpha Vantage
            if 'alpha_vantage_stocks' in self.sources:
                try:
                    data, meta_data = self.sources['alpha_vantage_stocks'].get_daily(symbol=symbol, outputsize='full')
                    # Im Cache speichern
                    if symbol not in self.data_cache['stocks']:
                        self.data_cache['stocks'][symbol] = {}
                    
                    self.data_cache['stocks'][symbol]['1d'] = data
                    return data.tail(limit)
                
                except Exception as fallback_error:
                    self.logger.error(f"Alpha Vantage Fallback fehlgeschlagen für {symbol}: {str(fallback_error)}")
            
            self.logger.warning(f"Keine Daten für {symbol} ({interval}) gefunden")
            return None
        
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Aktiendaten für {symbol} ({interval}): {str(e)}")
            return None

    def get_live_market_news(self, tickers: List[str] = None, limit: int = 10) -> List[Dict]:
        """
        Ruft aktuelle Marktnachrichten ab, gefiltert nach bestimmten Tickers.
        
        Args:
            tickers: Liste von Ticker-Symbolen oder Kategorien (optional)
            limit: Maximale Anzahl zurückgegebener Nachrichten
        
        Returns:
            Liste von Nachrichtenartikeln als Dictionaries
        """
        try:
            if not tickers:
                tickers = ['CRYPTO', 'BTC', 'ETH']
            
            # Kategorien für Nachrichten
            categories = ['crypto', 'stock', 'economy']
            all_news = []
            
            # Nachrichten aus dem Cache sammeln
            for category in categories:
                if category in self.data_cache['news']:
                    all_news.extend(self.data_cache['news'][category])
            
            # Falls keine Nachrichten im Cache, versuche sie abzurufen
            if not all_news:
                self._update_news_data()
                for category in categories:
                    if category in self.data_cache['news']:
                        all_news.extend(self.data_cache['news'][category])
            
            # Nach Tickers filtern
            tickers = [ticker.upper() for ticker in tickers]
            filtered_news = []
            
            for article in all_news:
                title = article.get('title', '').upper()
                description = article.get('description', '').upper()
                
                if any(ticker in title or ticker in description for ticker in tickers):
                    filtered_news.append(article)
            
            # Nach Veröffentlichungsdatum sortieren
            filtered_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
            
            return filtered_news[:limit]
        
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Marktnachrichten: {str(e)}")
            return []

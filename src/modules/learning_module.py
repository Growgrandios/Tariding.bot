# learning_module.py

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import datetime
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Machine Learning-Bibliotheken
import torch
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

# Technische Analyse
import talib
import talib.abstract as ta

class LearningModule:
    """
    Learning Module für den Trading Bot.
    
    Analysiert historische Daten, trainiert ML-Modelle und generiert
    Trading-Signale basierend auf Machine Learning.
    """
    
    def __init__(self, config: Dict[str, Any], data_pipeline=None):
        """
        Initialisiert das Learning Module.
        
        Args:
            config: Konfigurationseinstellungen
            data_pipeline: Optional, Referenz zur Datenpipeline
        """
        self.logger = logging.getLogger("LearningModule")
        self.logger.info("Initialisiere LearningModule...")
        
        # Konfiguration speichern
        self.config = config or {}
        
        # Basispfade für Modelle und Daten
        self.base_path = Path(self.config.get('base_path', 'data'))
        self.models_path = self.base_path / 'models'
        self.backtest_results_path = self.base_path / 'backtest_results'
        
        # Verzeichnisse erstellen falls nicht vorhanden
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.backtest_results_path.mkdir(parents=True, exist_ok=True)
        
        # Data Pipeline Referenz
        self.data_pipeline = data_pipeline
        
        # Trading-Parameter
        self.backtest_days = self.config.get('backtest_days', 90)
        self.paper_trading_days = self.config.get('paper_trading_days', 14)
        self.target_win_rate = self.config.get('target_win_rate', 0.6)
        
        # Training-Parameter
        training_config = self.config.get('training', {})
        self.epochs = training_config.get('epochs', 100)
        self.batch_size = training_config.get('batch_size', 32)
        self.validation_split = training_config.get('validation_split', 0.2)
        self.patience = training_config.get('patience', 10)
        
        # Liste der zu überwachenden Märkte
        self.markets = self.config.get('markets', [
            {'symbol': 'BTC/USDT', 'timeframes': ['15m', '1h', '4h']},
            {'symbol': 'ETH/USDT', 'timeframes': ['15m', '1h', '4h']},
            {'symbol': 'SOL/USDT', 'timeframes': ['1h', '4h']}
        ])
        
        # Geladene Modelle
        self.models = {}
        
        # Feature-Konfigurations-Manager
        self.feature_config = self._initialize_feature_config()
        
        # Handelshistorie
        self.trade_history = []
        
        # Performance-Metriken
        self.performance_metrics = {
            'trained_models': 0,
            'backtest_results': {},
            'global_accuracy': None,
            'last_training_time': None
        }
        
        # Multithreading-Schutz
        self.model_lock = threading.Lock()
        
        # Status
        self.is_training = False
        self.current_training_task = None
        
        self.logger.info("LearningModule erfolgreich initialisiert")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Setzt die Datenpipeline für den Zugriff auf Marktdaten.
        
        Args:
            data_pipeline: Referenz zur Datenpipeline
        """
        self.data_pipeline = data_pipeline
        self.logger.info("Datenpipeline erfolgreich verbunden")
    
    def _initialize_feature_config(self) -> Dict[str, Any]:
        """
        Initialisiert die Konfiguration für Feature-Generierung.
        
        Returns:
            Dictionary mit Feature-Konfiguration
        """
        # Standard-Feature-Konfiguration
        default_config = {
            'price_features': {
                'enabled': True,
                'ema': [8, 21, 55, 200],
                'sma': [10, 20, 50, 100],
                'returns': [1, 3, 5, 10],
                'log_returns': True
            },
            'volume_features': {
                'enabled': True,
                'vwap': True,
                'volume_ema': [8, 20],
                'volume_oscillators': True
            },
            'volatility_features': {
                'enabled': True,
                'atr_periods': [7, 14, 28],
                'bollinger_bands': {
                    'periods': [20],
                    'deviations': [2.0]
                }
            },
            'momentum_features': {
                'enabled': True,
                'rsi_periods': [7, 14, 21],
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9
                },
                'stochastic': {
                    'k': 14,
                    'd': 3,
                    'smooth': 3
                }
            },
            'pattern_features': {
                'enabled': True,
                'candlestick_patterns': True,
                'support_resistance': True
            },
            'market_features': {
                'enabled': True,
                'market_correlation': True,
                'sector_momentum': True
            }
        }
        
        # Benutzerdefinierte Konfiguration aus Config-Datei übernehmen
        user_config = self.config.get('feature_config', {})
        
        # Konfigurationen zusammenführen
        result_config = default_config.copy()
        self._merge_config_recursive(result_config, user_config)
        
        return result_config
    
    def _merge_config_recursive(self, base_config: Dict, override_config: Dict) -> None:
        """
        Führt zwei Konfigurationen rekursiv zusammen.
        
        Args:
            base_config: Basis-Konfiguration
            override_config: Überschreibende Konfiguration
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def preprocess_data(self, symbol: str, timeframe: str, days: int = None) -> pd.DataFrame:
        """
        Verarbeitet historische Daten für ein Symbol und einen Zeitrahmen.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1m', '5m', '15m', '1h', '4h', '1d')
            days: Anzahl der Tage für historische Daten (optional)
        
        Returns:
            DataFrame mit verarbeiteten Daten oder None bei Fehler
        """
        if not self.data_pipeline:
            self.logger.error("Keine Datenpipeline verfügbar")
            return None
        
        try:
            # Bestimme die Anzahl der benötigten Datenpunkte
            days_to_fetch = days or self.backtest_days
            
            # Heuristik für die Anzahl der Datenpunkte basierend auf Timeframe
            points_per_day = {
                '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
                '1h': 24, '2h': 12, '4h': 6, '6h': 4, '1d': 1
            }
            limit = days_to_fetch * (points_per_day.get(timeframe, 24) + 10)  # Extras für Sicherheit
            
            # Daten abrufen
            historical_data = self.data_pipeline.get_crypto_data(
                symbol, timeframe=timeframe, limit=limit
            )
            
            if historical_data is None or historical_data.empty:
                self.logger.warning(f"Keine Daten für {symbol} ({timeframe}) verfügbar")
                return None
            
            # Daten aufbereiten und Features berechnen
            processed_data = self._generate_features(historical_data)
            
            # Auf Vollständigkeit prüfen
            if processed_data is None or processed_data.empty:
                self.logger.warning(f"Verarbeitete Daten für {symbol} ({timeframe}) sind leer")
                return None
            
            # Filtern der Daten auf den gewünschten Zeitraum
            if days:
                start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                processed_data = processed_data[processed_data.index >= start_date]
            
            self.logger.info(f"Daten für {symbol} ({timeframe}) verarbeitet: {len(processed_data)} Datenpunkte")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenverarbeitung für {symbol} ({timeframe}): {str(e)}")
            return None
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generiert technische Indikatoren und Features aus den OHLCV-Daten.
        
        Args:
            df: DataFrame mit OHLCV-Daten
        
        Returns:
            DataFrame mit technischen Indikatoren und Features
        """
        if df is None or df.empty:
            return None
        
        try:
            # Tiefe Kopie um die Originaldaten nicht zu verändern
            result = df.copy()
            
            # Stelle sicher, dass alle erforderlichen Spalten vorhanden sind
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Erforderliche Spalten fehlen: {[col for col in required_columns if col not in df.columns]}")
                return None
            
            # 1. Preis-Features
            if self.feature_config['price_features']['enabled']:
                # Exponential Moving Averages
                for period in self.feature_config['price_features']['ema']:
                    result[f'ema_{period}'] = ta.EMA(result['close'], timeperiod=period)
                
                # Simple Moving Averages
                for period in self.feature_config['price_features']['sma']:
                    result[f'sma_{period}'] = ta.SMA(result['close'], timeperiod=period)
                
                # Relative Veränderungen (Returns)
                for period in self.feature_config['price_features']['returns']:
                    result[f'return_{period}'] = result['close'].pct_change(period)
                
                # Logarithmische Returns
                if self.feature_config['price_features']['log_returns']:
                    result['log_return_1'] = np.log(result['close'] / result['close'].shift(1))
            
            # 2. Volumen-Features
            if self.feature_config['volume_features']['enabled']:
                # Volume Weighted Average Price
                if self.feature_config['volume_features']['vwap']:
                    result['vwap'] = (result['volume'] * result['close']).cumsum() / result['volume'].cumsum()
                
                # Volume EMAs
                for period in self.feature_config['volume_features']['volume_ema']:
                    result[f'volume_ema_{period}'] = ta.EMA(result['volume'], timeperiod=period)
                
                # Volume Oszillatoren
                if self.feature_config['volume_features']['volume_oscillators']:
                    result['volume_change'] = result['volume'].pct_change()
                    result['volume_obv'] = ta.OBV(result['close'], result['volume'])
            
            # 3. Volatilitäts-Features
            if self.feature_config['volatility_features']['enabled']:
                # Average True Range
                for period in self.feature_config['volatility_features']['atr_periods']:
                    result[f'atr_{period}'] = ta.ATR(result['high'], result['low'], result['close'], timeperiod=period)
                
                # Bollinger Bands
                for period in self.feature_config['volatility_features']['bollinger_bands']['periods']:
                    for dev in self.feature_config['volatility_features']['bollinger_bands']['deviations']:
                        upper, middle, lower = ta.BBANDS(
                            result['close'], 
                            timeperiod=period, 
                            nbdevup=dev, 
                            nbdevdn=dev, 
                            matype=0
                        )
                        result[f'bb_upper_{period}_{int(dev)}'] = upper
                        result[f'bb_middle_{period}_{int(dev)}'] = middle
                        result[f'bb_lower_{period}_{int(dev)}'] = lower
                        # Prozentuale Lage im Band
                        result[f'bb_pct_{period}_{int(dev)}'] = (result['close'] - lower) / (upper - lower)
            
            # 4. Momentum-Features
            if self.feature_config['momentum_features']['enabled']:
                # Relative Strength Index
                for period in self.feature_config['momentum_features']['rsi_periods']:
                    result[f'rsi_{period}'] = ta.RSI(result['close'], timeperiod=period)
                
                # MACD
                macd_config = self.feature_config['momentum_features']['macd']
                macd, macd_signal, macd_hist = ta.MACD(
                    result['close'], 
                    fastperiod=macd_config['fast'], 
                    slowperiod=macd_config['slow'], 
                    signalperiod=macd_config['signal']
                )
                result['macd'] = macd
                result['macd_signal'] = macd_signal
                result['macd_hist'] = macd_hist
                
                # Stochastischer Oszillator
                stoch_config = self.feature_config['momentum_features']['stochastic']
                slowk, slowd = ta.STOCH(
                    result['high'], 
                    result['low'], 
                    result['close'], 
                    fastk_period=stoch_config['k'], 
                    slowk_period=stoch_config['smooth'], 
                    slowk_matype=0, 
                    slowd_period=stoch_config['d'], 
                    slowd_matype=0
                )
                result['stoch_k'] = slowk
                result['stoch_d'] = slowd
            
            # 5. Pattern-Features
            if self.feature_config['pattern_features']['enabled']:
                if self.feature_config['pattern_features']['candlestick_patterns']:
                    # Wichtige Candlestick-Muster
                    result['cdl_doji'] = ta.CDLDOJI(result['open'], result['high'], result['low'], result['close'])
                    result['cdl_hammer'] = ta.CDLHAMMER(result['open'], result['high'], result['low'], result['close'])
                    result['cdl_engulfing'] = ta.CDLENGULFING(result['open'], result['high'], result['low'], result['close'])
                    result['cdl_evening_star'] = ta.CDLEVENINGSTAR(result['open'], result['high'], result['low'], result['close'])
                    result['cdl_morning_star'] = ta.CDLMORNINGSTAR(result['open'], result['high'], result['low'], result['close'])
            
            # Ziel-Features für Prognosen hinzufügen (Zukünftige Preisänderungen)
            for forward_period in [1, 3, 6, 12, 24]:
                # Zukünftige absolute Preisänderung
                result[f'future_change_{forward_period}'] = result['close'].shift(-forward_period) - result['close']
                
                # Zukünftige prozentuale Preisänderung
                result[f'future_pct_{forward_period}'] = (result['close'].shift(-forward_period) / result['close'] - 1) * 100
                
                # Binäres Label (1 für Aufwärtstrend, 0 für Abwärtstrend)
                result[f'future_direction_{forward_period}'] = (result[f'future_pct_{forward_period}'] > 0).astype(int)
            
            # NaN-Werte entfernen
            result = result.dropna()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Feature-Generierung: {str(e)}")
            return None
    
    def train_model(self, symbol: str, timeframe: str, model_type: str = 'classification') -> bool:
        """
        Trainiert ein Modell für ein bestimmtes Symbol und einen Zeitrahmen.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('15m', '1h', '4h', '1d')
            model_type: Art des Modells ('classification' oder 'regression')
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        model_id = f"{symbol.replace('/', '_')}_{timeframe}_{model_type}"
        self.logger.info(f"Starte Training für Modell {model_id}...")
        
        try:
            # Markiere als in Training
            self.is_training = True
            self.current_training_task = model_id
            
            # Daten vorbereiten
            df = self.preprocess_data(symbol, timeframe, days=self.backtest_days)
            if df is None or df.empty:
                self.logger.error(f"Keine ausreichenden Daten für das Training von {model_id}")
                self.is_training = False
                return False
            
            # Features und Ziele extrahieren
            X, y = self._prepare_model_data(df, model_type)
            if X is None or y is None:
                self.logger.error(f"Fehler bei der Vorbereitung der Trainingsdaten für {model_id}")
                self.is_training = False
                return False
            
            # Daten aufteilen
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.validation_split, shuffle=False
            )
            
            # Modell initialisieren und trainieren
            model = self._create_model(model_type)
            if model is None:
                self.logger.error(f"Modell konnte nicht erstellt werden für {model_id}")
                self.is_training = False
                return False
            
            # Modell trainieren
            model.fit(X_train, y_train)
            
            # Modell evaluieren
            y_pred = model.predict(X_test)
            metrics = self._evaluate_model(y_test, y_pred, model_type)
            self.logger.info(f"Modell-Metriken für {model_id}: {metrics}")
            
            # Modell und Metadaten speichern
            self._save_model(model, model_id, {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': model_type,
                'metrics': metrics,
                'features': list(X.columns),
                'training_date': datetime.datetime.now().isoformat(),
                'training_samples': len(X_train)
            })
            
            # Performance-Metriken aktualisieren
            self.performance_metrics['trained_models'] += 1
            self.performance_metrics['last_training_time'] = datetime.datetime.now().isoformat()
            if 'accuracy' in metrics:
                if 'global_accuracy' not in self.performance_metrics or self.performance_metrics['global_accuracy'] is None:
                    self.performance_metrics['global_accuracy'] = metrics['accuracy']
                else:
                    # Gleitender Durchschnitt für die globale Genauigkeit
                    self.performance_metrics['global_accuracy'] = (
                        self.performance_metrics['global_accuracy'] * 0.7 + metrics['accuracy'] * 0.3
                    )
            
            self.is_training = False
            self.current_training_task = None
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training von {model_id}: {str(e)}")
            self.is_training = False
            self.current_training_task = None
            return False
    
    def _prepare_model_data(self, df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Bereitet Daten für das Modelltraining vor.
        
        Args:
            df: DataFrame mit Features
            model_type: Art des Modells ('classification' oder 'regression')
        
        Returns:
            Tuple aus Feature-DataFrame (X) und Ziel-Serie (y)
        """
        try:
            # Zielspaltennamen basierend auf Modelltyp
            if model_type == 'classification':
                target_column = 'future_direction_6'  # 6 Perioden Vorhersage (Richtung)
            else:
                target_column = 'future_pct_6'  # 6 Perioden Vorhersage (Prozentuale Änderung)
            
            if target_column not in df.columns:
                self.logger.error(f"Zielspalte {target_column} nicht in Daten gefunden")
                return None, None
            
            # Feature-Spalten auswählen (alle außer zukünftige Werte und einige andere)
            exclude_patterns = ['future_', 'timestamp', 'date']
            feature_columns = [col for col in df.columns if not any(pattern in col for pattern in exclude_patterns)]
            
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Datenvorbereitung: {str(e)}")
            return None, None
    
    def _create_model(self, model_type: str):
        """
        Erstellt ein neues Modell basierend auf dem Modelltyp.
        
        Args:
            model_type: Art des Modells ('classification' oder 'regression')
        
        Returns:
            Ein Modell-Objekt
        """
        try:
            if model_type == 'classification':
                # Klassifikationsmodell (Richtungsvorhersage)
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Regressionsmodell (Preisänderungsvorhersage)
                return GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Modellerstellung: {str(e)}")
            return None
    
    def _evaluate_model(self, y_true, y_pred, model_type: str) -> Dict[str, float]:
        """
        Evaluiert ein Modell und berechnet Leistungsmetriken.
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhergesagte Werte
            model_type: Art des Modells ('classification' oder 'regression')
        
        Returns:
            Dictionary mit Leistungsmetriken
        """
        try:
            if model_type == 'classification':
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0)
                }
            else:
                return {
                    'mse': mean_squared_error(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
                }
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Modellevaluierung: {str(e)}")
            return {'error': str(e)}
    
    def _save_model(self, model, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Speichert ein trainiertes Modell und seine Metadaten.
        
        Args:
            model: Das trainierte Modell
            model_id: Eindeutige ID für das Modell
            metadata: Metadaten für das Modell
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Thread-sicher speichern
            with self.model_lock:
                # Speicherpfade
                model_dir = self.models_path
                model_path = model_dir / f"{model_id}.pkl"
                metadata_path = model_dir / f"{model_id}_metadata.json"
                
                # Modell speichern
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Metadaten speichern
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Im Speicher halten
                self.models[model_id] = {
                    'model': model,
                    'metadata': metadata
                }
                
                self.logger.info(f"Modell {model_id} erfolgreich gespeichert")
                return True
                
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Modells {model_id}: {str(e)}")
            return False
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Lädt ein Modell aus der Datei.
        
        Args:
            model_id: Eindeutige ID für das Modell
        
        Returns:
            Dictionary mit Modell und Metadaten oder None bei Fehler
        """
        try:
            # Wenn Modell bereits geladen ist, gib es zurück
            if model_id in self.models:
                return self.models[model_id]
            
            # Thread-sicher laden
            with self.model_lock:
                # Dateipfade
                model_dir = self.models_path
                model_path = model_dir / f"{model_id}.pkl"
                metadata_path = model_dir / f"{model_id}_metadata.json"
                
                # Prüfen, ob Dateien existieren
                if not model_path.exists() or not metadata_path.exists():
                    self.logger.warning(f"Modell {model_id} nicht gefunden")
                    return None
                
                # Modell laden
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Metadaten laden
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Im Speicher halten
                self.models[model_id] = {
                    'model': model,
                    'metadata': metadata
                }
                
                self.logger.info(f"Modell {model_id} erfolgreich geladen")
                return self.models[model_id]
                
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells {model_id}: {str(e)}")
            return None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste aller verfügbaren Modelle und ihrer Metadaten zurück.
        
        Returns:
            Liste von Modellmetadaten
        """
        try:
            models_info = []
            
            # Durchsuche Modellverzeichnis nach Metadaten-Dateien
            metadata_files = list(self.models_path.glob("*_metadata.json"))
            
            for metadata_path in metadata_files:
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Modell-ID aus Dateinamen extrahieren
                    model_id = metadata_path.stem.replace('_metadata', '')
                    
                    # Information mit Metadaten und Verfügbarkeitsstatus
                    model_info = {
                        'model_id': model_id,
                        'loaded': model_id in self.models,
                        'metadata': metadata,
                        'file_path': str(metadata_path)
                    }
                    
                    models_info.append(model_info)
                except Exception as e:
                    self.logger.warning(f"Fehler beim Laden der Metadaten für {metadata_path}: {str(e)}")
            
            return models_info
            
        except Exception as e:
            self.logger.error(f"Fehler beim Auflisten verfügbarer Modelle: {str(e)}")
            return []
    
    def predict(self, symbol: str, timeframe: str, model_type: str = 'classification') -> Dict[str, Any]:
        """
        Führt eine Vorhersage für ein bestimmtes Symbol und einen Zeitrahmen durch.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('15m', '1h', '4h', '1d')
            model_type: Art des Modells ('classification' oder 'regression')
        
        Returns:
            Dictionary mit Vorhersagewerten und Konfidenz
        """
        try:
            # Modell-ID generieren
            model_id = f"{symbol.replace('/', '_')}_{timeframe}_{model_type}"
            
            # Modell laden
            model_data = self.load_model(model_id)
            if not model_data:
                self.logger.warning(f"Kein Modell für {model_id} gefunden, versuche ein neues zu trainieren")
                success = self.train_model(symbol, timeframe, model_type)
                if not success:
                    self.logger.error(f"Konnte kein Modell für {model_id} trainieren")
                    return {'error': 'Modell nicht verfügbar und konnte nicht trainiert werden'}
                model_data = self.load_model(model_id)
            
            # Aktuelle Marktdaten abrufen
            current_data = self.preprocess_data(symbol, timeframe, days=7)  # Letzte 7 Tage
            if current_data is None or current_data.empty:
                return {'error': 'Keine aktuellen Marktdaten verfügbar'}
            
            # Letzte Zeile für Prediction nehmen
            latest_data = current_data.iloc[-1:]
            
            # Features extrahieren, die im Modell verwendet wurden
            feature_names = model_data['metadata']['features']
            if not all(feature in latest_data.columns for feature in feature_names):
                missing = [f for f in feature_names if f not in latest_data.columns]
                self.logger.error(f"Fehlende Features in aktuellen Daten: {missing}")
                return {'error': 'Fehlende Features in aktuellen Daten'}
            
            X = latest_data[feature_names]
            
            # Vorhersage durchführen
            model = model_data['model']
            
            if model_type == 'classification':
                # Klassifikation (0 oder 1)
                prediction = model.predict(X)[0]
                # Wahrscheinlichkeiten
                probabilities = model.predict_proba(X)[0]
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
                
                result = {
                    'direction': 'up' if prediction == 1 else 'down',
                    'confidence': float(confidence),
                    'probability_up': float(probabilities[1]),
                    'probability_down': float(probabilities[0]),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            else:
                # Regression (prozentuale Preisänderung)
                prediction = model.predict(X)[0]
                
                result = {
                    'predicted_change_pct': float(prediction),
                    'direction': 'up' if prediction > 0 else 'down',
                    'timestamp': datetime.datetime.now().isoformat()
                }
            
            # Zusätzliche Informationen
            result['model_id'] = model_id
            result['model_type'] = model_type
            result['symbol'] = symbol
            result['timeframe'] = timeframe
            result['current_price'] = float(latest_data['close'].iloc[-1])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Vorhersage für {symbol} ({timeframe}): {str(e)}")
            return {'error': str(e)}
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Trainiert Modelle für alle konfigurierten Märkte und Zeitrahmen.
        
        Returns:
            Dictionary mit Trainingsergebnissen
        """
        self.logger.info("Starte Training für alle Modelle...")
        results = {
            'successful': [],
            'failed': [],
            'start_time': datetime.datetime.now().isoformat()
        }
        
        try:
            # Für jede Markt-Konfiguration
            for market in self.markets:
                symbol = market['symbol']
                timeframes = market.get('timeframes', ['1h', '4h'])
                
                for timeframe in timeframes:
                    # Klassifikationsmodell trainieren
                    model_id = f"{symbol.replace('/', '_')}_{timeframe}_classification"
                    self.logger.info(f"Training für {model_id}...")
                    
                    success = self.train_model(symbol, timeframe, 'classification')
                    if success:
                        results['successful'].append(model_id)
                    else:
                        results['failed'].append(model_id)
                    
                    # Optional: Regressionsmodell trainieren
                    if market.get('train_regression', False):
                        model_id = f"{symbol.replace('/', '_')}_{timeframe}_regression"
                        self.logger.info(f"Training für {model_id}...")
                        
                        success = self.train_model(symbol, timeframe, 'regression')
                        if success:
                            results['successful'].append(model_id)
                        else:
                            results['failed'].append(model_id)
            
            results['end_time'] = datetime.datetime.now().isoformat()
            results['total_models'] = len(results['successful']) + len(results['failed'])
            results['success_rate'] = len(results['successful']) / results['total_models'] if results['total_models'] > 0 else 0
            
            self.logger.info(f"Training abgeschlossen. Erfolgreiche Modelle: {len(results['successful'])}, Fehlgeschlagene: {len(results['failed'])}")
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training aller Modelle: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = datetime.datetime.now().isoformat()
            return results
    
    def run_backtest(self, symbol: str, timeframe: str, days: int = None, strategy: str = 'ml_basic') -> Dict[str, Any]:
        """
        Führt einen Backtest für ein bestimmtes Symbol und einen Zeitrahmen durch.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('15m', '1h', '4h', '1d')
            days: Anzahl der Tage für den Backtest (optional)
            strategy: Strategie-Typ für den Backtest
        
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        self.logger.info(f"Starte Backtest für {symbol} ({timeframe}) mit Strategie '{strategy}'...")
        
        try:
            # Daten für den Backtest abrufen
            backtest_days = days or self.backtest_days
            df = self.preprocess_data(symbol, timeframe, days=backtest_days)
            
            if df is None or df.empty:
                self.logger.error(f"Keine Daten für Backtest von {symbol} ({timeframe})")
                return {'error': 'Keine Daten verfügbar'}
            
            # Modell laden oder trainieren falls notwendig
            model_id = f"{symbol.replace('/', '_')}_{timeframe}_classification"
            model_data = self.load_model(model_id)
            
            if not model_data:
                self.logger.warning(f"Kein Modell für {model_id} gefunden, trainiere neu...")
                success = self.train_model(symbol, timeframe, 'classification')
                if not success:
                    self.logger.error(f"Konnte kein Modell für {model_id} trainieren")
                    return {'error': 'Modell konnte nicht trainiert werden'}
                model_data = self.load_model(model_id)
            
            # Features für das Modell extrahieren
            feature_names = model_data['metadata']['features']
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                self.logger.error(f"Fehlende Features für Backtest: {missing_features}")
                return {'error': f"Fehlende Features: {missing_features}"}
            
            # Modell und Daten für Backtest vorbereiten
            model = model_data['model']
            X = df[feature_names]
            
            # Vorhersagen für den gesamten Datensatz
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                df['predicted_proba'] = probabilities[:, 1]  # Wahrscheinlichkeit für Aufwärtsbewegung
            
            df['predicted_direction'] = model.predict(X)
            
            # Backtest-Strategie anwenden
            if strategy == 'ml_basic':
                results = self._backtest_ml_basic(df, symbol, timeframe)
            elif strategy == 'ml_with_indicators':
                results = self._backtest_ml_with_indicators(df, symbol, timeframe)
            else:
                self.logger.error(f"Unbekannte Backtest-Strategie: {strategy}")
                return {'error': f"Unbekannte Strategie: {strategy}"}
            
            # Ergebnisse speichern
            self._save_backtest_results(results, symbol, timeframe, strategy)
            
            # Aktualisiere Performance-Metriken
            if 'backtest_results' not in self.performance_metrics:
                self.performance_metrics['backtest_results'] = {}
            
            self.performance_metrics['backtest_results'][f"{symbol}_{timeframe}_{strategy}"] = {
                'win_rate': results['win_rate'],
                'total_return': results['total_return_pct'],
                'trades': results['total_trades']
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler beim Backtest für {symbol} ({timeframe}): {str(e)}")
            return {'error': str(e)}
    
    def _backtest_ml_basic(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Führt einen einfachen ML-basierten Backtest durch.
        
        Args:
            df: DataFrame mit Marktdaten und Vorhersagen
            symbol: Handelssymbol
            timeframe: Zeitrahmen
        
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        try:
            # Copy für Backtest
            backtest_df = df.copy()
            
            # Parameter
            entry_threshold = 0.65  # Min. Wahrscheinlichkeit für Einstieg
            take_profit_pct = 0.03  # Take-Profit (3%)
            stop_loss_pct = 0.015   # Stop-Loss (1.5%)
            
            # Handelslogik initialisieren
            backtest_df['position'] = 0
            backtest_df['entry_price'] = None
            backtest_df['exit_price'] = None
            backtest_df['trade_return'] = None
            backtest_df['cum_return'] = 1.0
            
            # Trades verfolgen
            trades = []
            in_position = False
            entry_price = 0
            entry_index = None
            position_type = None
            
            # Backtest durchführen
            for i in range(1, len(backtest_df)):
                current_price = backtest_df['close'].iloc[i]
                previous_close = backtest_df['close'].iloc[i-1]
                
                # Wenn wir nicht in einer Position sind, prüfe auf Einstiegssignale
                if not in_position:
                    # Long-Signal
                    if (backtest_df['predicted_proba'].iloc[i] > entry_threshold):
                        in_position = True
                        position_type = 'long'
                        entry_price = current_price
                        entry_index = backtest_df.index[i]
                        backtest_df.at[backtest_df.index[i], 'position'] = 1
                        backtest_df.at[backtest_df.index[i], 'entry_price'] = entry_price
                    
                    # Short-Signal
                    elif (backtest_df['predicted_proba'].iloc[i] < (1 - entry_threshold)):
                        in_position = True
                        position_type = 'short'
                        entry_price = current_price
                        entry_index = backtest_df.index[i]
                        backtest_df.at[backtest_df.index[i], 'position'] = -1
                        backtest_df.at[backtest_df.index[i], 'entry_price'] = entry_price
                
                # Wenn wir in einer Position sind, prüfe auf Ausstiegssignale
                elif in_position:
                    pct_change = (current_price / entry_price - 1) * (1 if position_type == 'long' else -1)
                    
                    # Take-Profit oder Stop-Loss erreicht
                    if (pct_change >= take_profit_pct) or (pct_change <= -stop_loss_pct):
                        # Position schließen
                        exit_price = current_price
                        trade_return = pct_change
                        
                        backtest_df.at[backtest_df.index[i], 'position'] = 0
                        backtest_df.at[backtest_df.index[i], 'exit_price'] = exit_price
                        backtest_df.at[backtest_df.index[i], 'trade_return'] = trade_return
                        
                        # Trade zur Liste hinzufügen
                        trade = {
                            'entry_time': entry_index,
                            'exit_time': backtest_df.index[i],
                            'position_type': position_type,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': trade_return * 100,
                            'exit_reason': 'take_profit' if pct_change >= take_profit_pct else 'stop_loss'
                        }
                        trades.append(trade)
                        
                        # Position zurücksetzen
                        in_position = False
                        entry_price = 0
                        entry_index = None
                        position_type = None
            
            # Offene Position am Ende des Backtests schließen
            if in_position:
                exit_price = backtest_df['close'].iloc[-1]
                pct_change = (exit_price / entry_price - 1) * (1 if position_type == 'long' else -1)
                trade_return = pct_change
                
                backtest_df.at[backtest_df.index[-1], 'position'] = 0
                backtest_df.at[backtest_df.index[-1], 'exit_price'] = exit_price
                backtest_df.at[backtest_df.index[-1], 'trade_return'] = trade_return
                
                # Trade zur Liste hinzufügen
                trade = {
                    'entry_time': entry_index,
                    'exit_time': backtest_df.index[-1],
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return * 100,
                    'exit_reason': 'end_of_backtest'
                }
                trades.append(trade)
            
            # Kumulative Returns berechnen
            trade_returns = [trade['return_pct'] / 100 for trade in trades]
            cumulative_return = 1.0
            for ret in trade_returns:
                cumulative_return *= (1 + ret)
            
            # Gewinn-/Verlusthandel-Statistik
            winning_trades = [t for t in trades if t['return_pct'] > 0]
            losing_trades = [t for t in trades if t['return_pct'] <= 0]
            
            # Ergebnisse berechnen
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': backtest_df.index[0].isoformat(),
                'end_date': backtest_df.index[-1].isoformat(),
                'initial_price': backtest_df['close'].iloc[0],
                'final_price': backtest_df['close'].iloc[-1],
                'market_return_pct': (backtest_df['close'].iloc[-1] / backtest_df['close'].iloc[0] - 1) * 100,
                'strategy_return_pct': (cumulative_return - 1) * 100,
                'total_return_pct': (cumulative_return - 1) * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_win_pct': np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss_pct': np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0,
                'max_drawdown_pct': self._calculate_max_drawdown(trade_returns) * 100,
                'sharpe_ratio': self._calculate_sharpe_ratio(trade_returns),
                'trades': trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler im ML-Basic Backtest: {str(e)}")
            return {'error': str(e)}
    
    def _backtest_ml_with_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Führt einen ML-basierten Backtest mit zusätzlichen technischen Indikatoren durch.
        
        Args:
            df: DataFrame mit Marktdaten und Vorhersagen
            symbol: Handelssymbol
            timeframe: Zeitrahmen
        
        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        try:
            # Copy für Backtest
            backtest_df = df.copy()
            
            # Parameter
            entry_threshold = 0.60  # Min. Wahrscheinlichkeit für Einstieg
            take_profit_pct = 0.04  # Take-Profit (4%)
            stop_loss_pct = 0.02    # Stop-Loss (2%)
            
            # Handelslogik initialisieren
            backtest_df['position'] = 0
            backtest_df['entry_price'] = None
            backtest_df['exit_price'] = None
            backtest_df['trade_return'] = None
            backtest_df['cum_return'] = 1.0
            
            # Trades verfolgen
            trades = []
            in_position = False
            entry_price = 0
            entry_index = None
            position_type = None
            
            # Backtest durchführen
            for i in range(1, len(backtest_df)):
                current_price = backtest_df['close'].iloc[i]
                previous_close = backtest_df['close'].iloc[i-1]
                
                # Zusätzliche Indikatoren prüfen
                rsi_oversold = False
                rsi_overbought = False
                if 'rsi_14' in backtest_df.columns:
                    rsi = backtest_df['rsi_14'].iloc[i]
                    rsi_oversold = rsi < 30
                    rsi_overbought = rsi > 70
                
                trend_up = False
                trend_down = False
                if 'ema_55' in backtest_df.columns and 'ema_21' in backtest_df.columns:
                    ema_short = backtest_df['ema_21'].iloc[i]
                    ema_long = backtest_df['ema_55'].iloc[i]
                    trend_up = ema_short > ema_long
                    trend_down = ema_short < ema_long
                
                # Wenn wir nicht in einer Position sind, prüfe auf Einstiegssignale
                if not in_position:
                    # Long-Signal
                    if (backtest_df['predicted_proba'].iloc[i] > entry_threshold and
                        (trend_up or rsi_oversold)):
                        in_position = True
                        position_type = 'long'
                        entry_price = current_price
                        entry_index = backtest_df.index[i]
                        backtest_df.at[backtest_df.index[i], 'position'] = 1
                        backtest_df.at[backtest_df.index[i], 'entry_price'] = entry_price
                    
                    # Short-Signal
                    elif (backtest_df['predicted_proba'].iloc[i] < (1 - entry_threshold) and
                          (trend_down or rsi_overbought)):
                        in_position = True
                        position_type = 'short'
                        entry_price = current_price
                        entry_index = backtest_df.index[i]
                        backtest_df.at[backtest_df.index[i], 'position'] = -1
                        backtest_df.at[backtest_df.index[i], 'entry_price'] = entry_price
                
                # Wenn wir in einer Position sind, prüfe auf Ausstiegssignale
                elif in_position:
                    pct_change = (current_price / entry_price - 1) * (1 if position_type == 'long' else -1)
                    
                    # Take-Profit oder Stop-Loss erreicht oder Trendumkehr
                    exit_condition = (
                        (pct_change >= take_profit_pct) or 
                        (pct_change <= -stop_loss_pct) or
                        (position_type == 'long' and backtest_df['predicted_proba'].iloc[i] < 0.4) or
                        (position_type == 'short' and backtest_df['predicted_proba'].iloc[i] > 0.6)
                    )
                    
                    if exit_condition:
                        # Position schließen
                        exit_price = current_price
                        trade_return = pct_change
                        
                        backtest_df.at[backtest_df.index[i], 'position'] = 0
                        backtest_df.at[backtest_df.index[i], 'exit_price'] = exit_price
                        backtest_df.at[backtest_df.index[i], 'trade_return'] = trade_return
                        
                        # Exit-Grund bestimmen
                        if pct_change >= take_profit_pct:
                            exit_reason = 'take_profit'
                        elif pct_change <= -stop_loss_pct:
                            exit_reason = 'stop_loss'
                        else:
                            exit_reason = 'signal_reversal'
                        
                        # Trade zur Liste hinzufügen
                        trade = {
                            'entry_time': entry_index,
                            'exit_time': backtest_df.index[i],
                            'position_type': position_type,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': trade_return * 100,
                            'exit_reason': exit_reason
                        }
                        trades.append(trade)
                        
                        # Position zurücksetzen
                        in_position = False
                        entry_price = 0
                        entry_index = None
                        position_type = None
            
            # Offene Position am Ende des Backtests schließen
            if in_position:
                exit_price = backtest_df['close'].iloc[-1]
                pct_change = (exit_price / entry_price - 1) * (1 if position_type == 'long' else -1)
                trade_return = pct_change
                
                backtest_df.at[backtest_df.index[-1], 'position'] = 0
                backtest_df.at[backtest_df.index[-1], 'exit_price'] = exit_price
                backtest_df.at[backtest_df.index[-1], 'trade_return'] = trade_return
                
                # Trade zur Liste hinzufügen
                trade = {
                    'entry_time': entry_index,
                    'exit_time': backtest_df.index[-1],
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return * 100,
                    'exit_reason': 'end_of_backtest'
                }
                trades.append(trade)
            
            # Kumulative Returns berechnen
            trade_returns = [trade['return_pct'] / 100 for trade in trades]
            cumulative_return = 1.0
            for ret in trade_returns:
                cumulative_return *= (1 + ret)
            
            # Gewinn-/Verlusthandel-Statistik
            winning_trades = [t for t in trades if t['return_pct'] > 0]
            losing_trades = [t for t in trades if t['return_pct'] <= 0]
            
            # Ergebnisse berechnen
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': backtest_df.index[0].isoformat(),
                'end_date': backtest_df.index[-1].isoformat(),
                'initial_price': backtest_df['close'].iloc[0],
                'final_price': backtest_df['close'].iloc[-1],
                'market_return_pct': (backtest_df['close'].iloc[-1] / backtest_df['close'].iloc[0] - 1) * 100,
                'strategy_return_pct': (cumulative_return - 1) * 100,
                'total_return_pct': (cumulative_return - 1) * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_win_pct': np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss_pct': np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0,
                'max_drawdown_pct': self._calculate_max_drawdown(trade_returns) * 100,
                'sharpe_ratio': self._calculate_sharpe_ratio(trade_returns),
                'trades': trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler im ML-with-Indicators Backtest: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, returns) -> float:
        """
        Berechnet den maximalen Drawdown aus einer Liste von Returns.
        
        Args:
            returns: Liste von prozentualen Returns
        
        Returns:
            Max. Drawdown als Dezimalwert
        """
        if not returns:
            return 0.0
        
        try:
            # Kumulative Returns berechnen
            cum_returns = np.cumprod(np.array([1 + ret for ret in returns]))
            # Running maximum berechnen
            running_max = np.maximum.accumulate(cum_returns)
            # Drawdown berechnen
            drawdown = (running_max - cum_returns) / running_max
            # Maximalen Drawdown zurückgeben
            return np.max(drawdown)
        except Exception as e:
            self.logger.error(f"Fehler bei der Drawdown-Berechnung: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0, periods_per_year=252) -> float:
        """
        Berechnet das Sharpe-Verhältnis.
        
        Args:
            returns: Liste von prozentualen Returns
            risk_free_rate: Risikoloser Zinssatz (Default: 0)
            periods_per_year: Anzahl der Perioden pro Jahr
        
        Returns:
            Sharpe-Verhältnis
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        try:
            # Annualisierte Returns berechnen
            mean_return = np.mean(returns)
            annual_return = (1 + mean_return) ** periods_per_year - 1
            
            # Annualisierte Standardabweichung
            std_return = np.std(returns, ddof=1)
            annual_std = std_return * np.sqrt(periods_per_year)
            
            # Sharpe-Verhältnis
            if annual_std == 0:
                return 0.0
            sharpe = (annual_return - risk_free_rate) / annual_std
            
            return sharpe
        except Exception as e:
            self.logger.error(f"Fehler bei der Sharpe-Ratio-Berechnung: {str(e)}")
            return 0.0
    
    def _save_backtest_results(self, results: Dict[str, Any], symbol: str, timeframe: str, strategy: str) -> bool:
        """
        Speichert die Ergebnisse eines Backtests.
        
        Args:
            results: Backtest-Ergebnisse
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            strategy: Strategie-Typ
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Dateipfad generieren
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{strategy}_{timestamp}.json"
            file_path = self.backtest_results_path / filename
            
            # Ergebnisse speichern
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Backtest-Ergebnisse gespeichert: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Backtest-Ergebnisse: {str(e)}")
            return False
    
    def get_backtest_results(self, symbol: str = None, timeframe: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gibt die Ergebnisse früherer Backtests zurück.
        
        Args:
            symbol: Optional, filter nach Symbol
            timeframe: Optional, filter nach Zeitrahmen
            limit: Maximale Anzahl zurückzugebender Ergebnisse
        
        Returns:
            Liste von Backtest-Ergebnissen
        """
        try:
            results = []
            
            # Alle JSON-Dateien im Backtest-Verzeichnis
            json_files = list(self.backtest_results_path.glob("*.json"))
            
            # Nach neuesten sortieren
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Dateien durchlaufen
            for file_path in json_files:
                filename = file_path.name
                
                # Filter anwenden
                if symbol and symbol.replace('/', '_') not in filename:
                    continue
                if timeframe and timeframe not in filename:
                    continue
                
                # Ergebnisse laden
                try:
                    with open(file_path, 'r') as f:
                        backtest_data = json.load(f)
                    
                    # Dateipfad hinzufügen
                    backtest_data['file_path'] = str(file_path)
                    
                    results.append(backtest_data)
                    
                    # Limit prüfen
                    if len(results) >= limit:
                        break
                        
                except Exception as file_error:
                    self.logger.warning(f"Fehler beim Laden der Backtest-Datei {file_path}: {str(file_error)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Backtest-Ergebnisse: {str(e)}")
            return []
    
    def get_market_signals(self) -> Dict[str, Any]:
        """
        Generiert Trading-Signale für alle konfigurierten Märkte.
        
        Returns:
            Dictionary mit Handelsempfehlungen
        """
        try:
            signals = {
                'timestamp': datetime.datetime.now().isoformat(),
                'signals': [],
                'summary': {
                    'bullish': 0,
                    'bearish': 0,
                    'neutral': 0
                }
            }
            
            for market in self.markets:
                symbol = market['symbol']
                timeframes = market.get('timeframes', ['1h', '4h'])
                
                for timeframe in timeframes:
                    signal = self.get_market_signal(symbol, timeframe)
                    signals['signals'].append(signal)
                    
                    # Zusammenfassung aktualisieren
                    if 'signal' in signal:
                        sig_type = signal['signal']
                        if sig_type == 'buy':
                            signals['summary']['bullish'] += 1
                        elif sig_type == 'sell':
                            signals['summary']['bearish'] += 1
                        else:
                            signals['summary']['neutral'] += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Fehler beim Generieren der Marktsignale: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.datetime.now().isoformat()}
    
    def get_market_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Generiert ein Handelssignal für ein bestimmtes Symbol und einen Zeitrahmen.
        
        Args:
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('15m', '1h', '4h', '1d')
        
        Returns:
            Dictionary mit Handelssignal und ergänzenden Informationen
        """
        try:
            # ML-Vorhersage abrufen
            prediction = self.predict(symbol, timeframe, 'classification')
            
            if 'error' in prediction:
                self.logger.warning(f"Fehler bei der ML-Vorhersage für {symbol} ({timeframe}): {prediction['error']}")
                # Fallback zu traditioneller Analyse
                return self._generate_traditional_signal(symbol, timeframe)
            
            # Aktuelle Marktdaten abrufen
            current_data = self.preprocess_data(symbol, timeframe, days=7)  # Letzte 7 Tage
            if current_data is None or current_data.empty:
                self.logger.warning(f"Keine aktuellen Marktdaten für {symbol} ({timeframe})")
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': 'neutral',
                    'confidence': 0,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'message': 'Keine aktuellen Marktdaten verfügbar'
                }
            
            # Technische Indikatoren erfassen
            latest_data = current_data.iloc[-1]
            current_price = float(latest_data['close'])
            
            # RSI für Überkauft/Überverkauft-Indikation
            rsi = float(latest_data['rsi_14']) if 'rsi_14' in latest_data else None
            
            # MACD für Trendstärke
            macd = float(latest_data['macd']) if 'macd' in latest_data else None
            macd_signal = float(latest_data['macd_signal']) if 'macd_signal' in latest_data else None
            macd_hist = float(latest_data['macd_hist']) if 'macd_hist' in latest_data else None
            
            # Bollinger-Bänder für Volatilität
            bb_upper = float(latest_data['bb_upper_20_2']) if 'bb_upper_20_2' in latest_data else None
            bb_lower = float(latest_data['bb_lower_20_2']) if 'bb_lower_20_2' in latest_data else None
            bb_width = ((bb_upper - bb_lower) / current_price) if bb_upper and bb_lower else None
            
            # Kombiniertes Signal generieren
            ml_direction = prediction['direction']
            ml_confidence = prediction['confidence']
            
            # Signal-Logik
            signal = 'neutral'
            signal_strength = 0.0
            reasons = []
            
            # ML-Vorhersage als Hauptsignal
            if ml_direction == 'up' and ml_confidence > 0.6:
                signal = 'buy'
                signal_strength = ml_confidence
                reasons.append(f"ML-Modell prognostiziert Aufwärtsbewegung (Konf.: {ml_confidence:.2f})")
            elif ml_direction == 'down' and ml_confidence > 0.6:
                signal = 'sell'
                signal_strength = ml_confidence
                reasons.append(f"ML-Modell prognostiziert Abwärtsbewegung (Konf.: {ml_confidence:.2f})")
            else:
                reasons.append(f"ML-Modell unentschieden (Richtung: {ml_direction}, Konf.: {ml_confidence:.2f})")
            
            # RSI zur Bestätigung
            if rsi is not None:
                if rsi < 30:
                    reasons.append(f"RSI überverkauft ({rsi:.2f})")
                    if signal == 'neutral':
                        signal = 'buy'
                        signal_strength = 0.6
                    elif signal == 'buy':
                        signal_strength = min(0.9, signal_strength + 0.1)
                elif rsi > 70:
                    reasons.append(f"RSI überkauft ({rsi:.2f})")
                    if signal == 'neutral':
                        signal = 'sell'
                        signal_strength = 0.6
                    elif signal == 'sell':
                        signal_strength = min(0.9, signal_strength + 0.1)
            
            # MACD zur Bestätigung
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    reasons.append("MACD über Signallinie (bullish)")
                    if signal == 'buy':
                        signal_strength = min(0.95, signal_strength + 0.05)
                elif macd < macd_signal:
                    reasons.append("MACD unter Signallinie (bearish)")
                    if signal == 'sell':
                        signal_strength = min(0.95, signal_strength + 0.05)
            
            # Trend-Analyse (EMA-Kreuzung)
            ema_short = float(latest_data['ema_8']) if 'ema_8' in latest_data else None
            ema_long = float(latest_data['ema_21']) if 'ema_21' in latest_data else None
            
            if ema_short is not None and ema_long is not None:
                if ema_short > ema_long:
                    reasons.append("Kurzfristiger EMA über langfristigem EMA (bullish)")
                    if signal == 'buy':
                        signal_strength = min(0.95, signal_strength + 0.05)
                elif ema_short < ema_long:
                    reasons.append("Kurzfristiger EMA unter langfristigem EMA (bearish)")
                    if signal == 'sell':
                        signal_strength = min(0.95, signal_strength + 0.05)
            
            # Signalstärke auf Konfidenz mappen
            confidence = signal_strength
            
            # Signal-Nachricht generieren
            if signal == 'buy':
                message = f"Kaufsignal für {symbol} mit {confidence:.0%} Konfidenz"
            elif signal == 'sell':
                message = f"Verkaufssignal für {symbol} mit {confidence:.0%} Konfidenz"
            else:
                message = f"Neutrales Signal für {symbol}"
            
            # Vollständiges Signal-Objekt zusammenstellen
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'ml_direction': ml_direction,
                'ml_confidence': ml_confidence,
                'timestamp': datetime.datetime.now().isoformat(),
                'message': message,
                'reasons': reasons,
                'indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_hist': macd_hist,
                    'bb_width': bb_width
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler beim Generieren des Marktsignals für {symbol} ({timeframe}): {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'error',
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_traditional_signal(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generiert ein Handelssignal basierend auf traditioneller technischer Analyse.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
        
        Returns:
            Dictionary mit Handelssignal
        """
        try:
            # Marktdaten abrufen
            df = self.preprocess_data(symbol, timeframe, days=7)
            
            if df is None or df.empty:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': 'neutral',
                    'confidence': 0,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'message': 'Keine Daten verfügbar für traditionelle Analyse'
                }
            
            # Letzte Daten extrahieren
            latest = df.iloc[-1]
            
            # Signale prüfen
            buy_signals = 0
            sell_signals = 0
            reasons = []
            
            # 1. RSI-Signal
            if 'rsi_14' in latest:
                rsi = latest['rsi_14']
                if rsi < 30:
                    buy_signals += 2
                    reasons.append(f"RSI überverkauft ({rsi:.2f})")
                elif rsi > 70:
                    sell_signals += 2
                    reasons.append(f"RSI überkauft ({rsi:.2f})")
            
            # 2. MACD-Signal
            if 'macd' in latest and 'macd_signal' in latest:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                if macd > macd_signal:
                    buy_signals += 1
                    reasons.append("MACD über Signallinie (bullish)")
                elif macd < macd_signal:
                    sell_signals += 1
                    reasons.append("MACD unter Signallinie (bearish)")
            
            # 3. Bollinger Bands
            if 'bb_upper_20_2' in latest and 'bb_lower_20_2' in latest:
                close = latest['close']
                bb_upper = latest['bb_upper_20_2']
                bb_lower = latest['bb_lower_20_2']
                
                if close >= bb_upper:
                    sell_signals += 1
                    reasons.append("Preis am oberen Bollinger-Band (mögliche Überkauft-Situation)")
                elif close <= bb_lower:
                    buy_signals += 1
                    reasons.append("Preis am unteren Bollinger-Band (mögliche Überverkauft-Situation)")
            
            # 4. Trend basierend auf EMAs
            if 'ema_8' in latest and 'ema_21' in latest:
                ema_short = latest['ema_8']
                ema_long = latest['ema_21']
                
                if ema_short > ema_long:
                    buy_signals += 1
                    reasons.append("Kurzfristiger EMA über langfristigem EMA (bullish)")
                elif ema_short < ema_long:
                    sell_signals += 1
                    reasons.append("Kurzfristiger EMA unter langfristigem EMA (bearish)")
            
            # Signal berechnen
            signal = 'neutral'
            if buy_signals > sell_signals + 1:
                signal = 'buy'
            elif sell_signals > buy_signals + 1:
                signal = 'sell'
            
            # Konfidenz berechnen
            total_signals = buy_signals + sell_signals
            signal_diff = abs(buy_signals - sell_signals)
            confidence = min(0.9, signal_diff / max(4, total_signals) if total_signals > 0 else 0)
            
            # Nachricht generieren
            if signal == 'buy':
                message = f"Kaufsignal für {symbol} basierend auf technischer Analyse"
            elif signal == 'sell':
                message = f"Verkaufssignal für {symbol} basierend auf technischer Analyse"
            else:
                message = f"Neutrales Signal für {symbol}"
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.datetime.now().isoformat(),
                'message': message,
                'reasons': reasons,
                'method': 'traditional',
                'buy_signals': buy_signals,
                'sell_signals': sell_signals
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der traditionellen Signalgenerierung für {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'error',
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Gibt Statusinformationen zum Learning Module zurück.
        
        Returns:
            Dictionary mit aktuellen Statusinformationen
        """
        try:
            # Modelle zählen
            model_files = list(self.models_path.glob("*.pkl"))
            
            # Überprüfe neuste Trainings- und Backtestdateien
            backtest_files = list(self.backtest_results_path.glob("*.json"))
            backtest_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Neuster Backtest
            latest_backtest = None
            if backtest_files:
                try:
                    with open(backtest_files[0], 'r') as f:
                        latest_backtest = json.load(f)
                except:
                    pass
            
            # Status konstruieren
            status = {
                'is_training': self.is_training,
                'current_training_task': self.current_training_task,
                'available_models': len(model_files),
                'loaded_models': len(self.models),
                'performance_metrics': self.performance_metrics,
                'latest_backtest': latest_backtest,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des aktuellen Status: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }

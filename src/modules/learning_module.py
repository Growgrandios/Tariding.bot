# learning_module.py

import os
import logging
import threading
import time
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/learning_module.log"),
        logging.StreamHandler()
    ]
)

class LearningModule:
    """
    Modul für Machine Learning und Backtesting von Handelsstrategien.
    Trainiert verschiedene Modelle auf historischen Daten und bewertet ihre Performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert das Learning-Modul.
        
        Args:
            config: Konfigurationseinstellungen für das Learning-Modul
        """
        self.logger = logging.getLogger("LearningModule")
        self.logger.info("Initialisiere LearningModule...")
        
        # Konfiguration laden
        self.config = config
        self.backtest_days = config.get('backtest_days', 90)
        self.paper_trading_days = config.get('paper_trading_days', 14)
        self.target_win_rate = config.get('target_win_rate', 0.6)
        
        # Trainingskonfiguration
        self.training_config = config.get('training', {})
        self.epochs = self.training_config.get('epochs', 100)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.validation_split = self.training_config.get('validation_split', 0.2)
        self.patience = self.training_config.get('patience', 10)
        
        # Pfade für Modelle und Ergebnisse
        self.models_path = Path('data/models')
        self.results_path = Path('data/backtest_results')
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Modelle und Ergebnisse
        self.models = {}
        self.performance_metrics = {}
        self.current_signals = {}
        self.trade_history = []
        
        # Features
        self.default_features = [
            'close', 'volume', 'rsi_14', 'ema_9', 'ema_21', 'macd', 'macd_signal',
            'bband_upper', 'bband_lower', 'atr_14'
        ]
        
        # Daten Pipeline (wird später gesetzt)
        self.data_pipeline = None
        
        # Thread-Management
        self.is_training = False
        self.is_backtesting = False
        self.training_thread = None
        self.backtesting_thread = None
        
        self.logger.info("LearningModule erfolgreich initialisiert")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Setzt die Datenpipeline für den Zugriff auf Marktdaten.
        
        Args:
            data_pipeline: Referenz zur Datenpipeline
        """
        self.data_pipeline = data_pipeline
        self.logger.info("Datenpipeline erfolgreich verbunden")
    
    def train_model(self, model_name: str, symbol: str, timeframe: str = '1h',
                  features: Optional[List[str]] = None, start_date: Optional[str] = None,
                  end_date: Optional[str] = None, model_type: str = 'random_forest',
                  target_metric: str = 'direction'):
        """
        Trainiert ein Machine-Learning-Modell für ein bestimmtes Symbol und Zeitrahmen.
        
        Args:
            model_name: Name des Modells für die spätere Referenzierung
            symbol: Handelssymbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen ('1h', '4h', '1d', etc.)
            features: Liste von Features für das Training (optional)
            start_date: Startdatum für Trainingsdaten (optional)
            end_date: Enddatum für Trainingsdaten (optional)
            model_type: Typ des Modells ('random_forest', 'gradient_boosting', 'lstm')
            target_metric: Zielmetrik ('direction', 'volatility', 'return')
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.is_training:
            self.logger.warning(f"Training für {model_name} kann nicht gestartet werden, da bereits ein Training läuft")
            return False
        
        if not self.data_pipeline:
            self.logger.error("Keine Datenpipeline verfügbar, Training nicht möglich")
            return False
        
        # Features festlegen
        if features is None:
            features = self.default_features
        
        # Daten vorbereiten
        try:
            # Zeitraum festlegen, falls nicht angegeben
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                # Start-Datum basierend auf Backtest-Tagen berechnen
                start_date = (datetime.now() - timedelta(days=self.backtest_days)).strftime('%Y-%m-%d')
            
            # Training in separatem Thread starten
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self._train_model_thread,
                args=(model_name, symbol, timeframe, features, start_date, end_date, model_type, target_metric),
                daemon=True
            )
            self.training_thread.start()
            
            self.logger.info(f"Training für Modell {model_name} gestartet (Symbol: {symbol}, Timeframe: {timeframe})")
            return True
        
        except Exception as e:
            self.logger.error(f"Fehler beim Starten des Trainings für {model_name}: {str(e)}")
            self.is_training = False
            return False
    
    def _train_model_thread(self, model_name: str, symbol: str, timeframe: str,
                          features: List[str], start_date: str, end_date: str,
                          model_type: str, target_metric: str):
        """
        Thread-Funktion für das Training eines Modells.
        
        Args:
            model_name: Name des Modells
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            features: Liste von Features
            start_date: Startdatum
            end_date: Enddatum
            model_type: Typ des Modells
            target_metric: Zielmetrik
        """
        try:
            # Daten laden
            df = self._load_and_prepare_data(symbol, timeframe, start_date, end_date)
            
            if df is None or df.empty:
                self.logger.error(f"Keine Daten für {symbol} ({timeframe}) verfügbar")
                self.is_training = False
                return
            
            # Features und Zielvariable vorbereiten
            X, y = self._prepare_features_and_target(df, features, target_metric)
            
            if X is None or y is None:
                self.logger.error(f"Fehler bei der Feature-Vorbereitung für {symbol}")
                self.is_training = False
                return
            
            # Trainings- und Testdaten aufteilen
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split, shuffle=False)
            
            # Modell trainieren
            model, train_metrics = self._train_specific_model(model_type, X_train, y_train, X_test, y_test)
            
            if model is None:
                self.logger.error(f"Fehler beim Training des Modells {model_name}")
                self.is_training = False
                return
            
            # Modell speichern
            self._save_model(model_name, model, model_type)
            
            # Performance-Metriken speichern
            self.performance_metrics[model_name] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': model_type,
                'target': target_metric,
                'features': features,
                'train_metrics': train_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info(f"Training für {model_name} abgeschlossen. Metriken: {train_metrics}")
            
            # Backtesting durchführen
            self.backtest_model(model_name)
            
        except Exception as e:
            self.logger.error(f"Fehler im Training Thread für {model_name}: {str(e)}")
        
        finally:
            self.is_training = False
    
    def _load_and_prepare_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Lädt und bereitet Daten für das Training vor.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            start_date: Startdatum
            end_date: Enddatum
            
        Returns:
            DataFrame mit vorbereiteten Daten oder None bei Fehler
        """
        try:
            # Daten über die Datenpipeline laden
            df = self.data_pipeline.fetch_crypto_data(symbol, start_date, end_date, timeframe)
            
            if df is None or df.empty:
                self.logger.error(f"Keine Daten für {symbol} im Zeitraum {start_date} bis {end_date} verfügbar")
                return None
            
            # Technische Indikatoren hinzufügen
            indicators = [
                {'type': 'rsi', 'period': 14},
                {'type': 'ema', 'period': 9},
                {'type': 'ema', 'period': 21},
                {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                {'type': 'bollinger', 'period': 20, 'std_dev': 2},
                {'type': 'atr', 'period': 14}
            ]
            
            df = self.data_pipeline.get_indicators(df, indicators)
            
            # NaN-Werte entfernen
            df = df.dropna()
            
            # Zukünftige Preisänderungen berechnen (für die Zielwerte)
            df['next_close'] = df['close'].shift(-1)
            df['price_change'] = df['next_close'] - df['close']
            df['returns'] = df['price_change'] / df['close']
            df['direction'] = (df['returns'] > 0).astype(int)
            
            # Volatilität berechnen
            df['volatility'] = df['close'].pct_change().rolling(window=5).std()
            
            # Extremwerte entfernen
            for col in df.columns:
                if col in ['timestamp', 'direction']:
                    continue
                
                q_low = df[col].quantile(0.005)
                q_high = df[col].quantile(0.995)
                df[col] = df[col].clip(q_low, q_high)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden und Vorbereiten der Daten für {symbol}: {str(e)}")
            return None
    
    def _prepare_features_and_target(self, df: pd.DataFrame, features: List[str], target_metric: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Bereitet Features und Zielvariable aus dem DataFrame vor.
        
        Args:
            df: DataFrame mit Daten
            features: Liste von Feature-Spalten
            target_metric: Zielmetrik ('direction', 'volatility', 'return')
            
        Returns:
            Tuple aus Feature-Matrix und Ziel-Array oder (None, None) bei Fehler
        """
        try:
            # Prüfen, ob alle Features im DataFrame vorhanden sind
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                self.logger.error(f"Fehlende Features im DataFrame: {missing_features}")
                return None, None
            
            # Features extrahieren
            X = df[features].values
            
            # Feature-Normalisierung
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Zielvariable extrahieren
            if target_metric == 'direction':
                y = df['direction'].values
            elif target_metric == 'volatility':
                # Diskretisiere Volatilität in Klassen (hoch/niedrig)
                volatility_threshold = df['volatility'].quantile(0.7)
                y = (df['volatility'] > volatility_threshold).astype(int).values
            elif target_metric == 'return':
                # Diskretisiere Returns in Klassen
                y = pd.qcut(df['returns'], q=3, labels=[0, 1, 2]).astype(int).values
            else:
                self.logger.error(f"Ungültige Zielmetrik: {target_metric}")
                return None, None
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Feature-Vorbereitung: {str(e)}")
            return None, None
    
    def _train_specific_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """
        Trainiert ein spezifisches Modell basierend auf dem angegebenen Typ.
        
        Args:
            model_type: Typ des Modells ('random_forest', 'gradient_boosting', 'lstm')
            X_train: Trainings-Features
            y_train: Trainings-Ziele
            X_test: Test-Features
            y_test: Test-Ziele
            
        Returns:
            Tuple aus trainiertem Modell und Metrik-Dictionary
        """
        model = None
        metrics = {}
        
        try:
            if model_type == 'random_forest':
                # RandomForest-Modell trainieren
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # Vorhersagen und Metriken berechnen
                y_pred = model.predict(X_test)
                
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                
                # Feature-Importance
                metrics['feature_importance'] = model.feature_importances_.tolist()
                
            elif model_type == 'gradient_boosting':
                # GradientBoosting-Modell trainieren
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Vorhersagen und Metriken berechnen
                y_pred = model.predict(X_test)
                
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                
                # Feature-Importance
                metrics['feature_importance'] = model.feature_importances_.tolist()
                
            elif model_type == 'lstm':
                # LSTM-Modell erstellen
                input_shape = (X_train.shape[1], 1)
                X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=input_shape),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Callbacks für das Training
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
                
                # Modell trainieren
                history = model.fit(
                    X_train_reshaped,
                    y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1
                )
                
                # Vorhersagen und Metriken berechnen
                y_pred_proba = model.predict(X_test_reshaped)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                
                # Trainingsverlauf
                metrics['training_history'] = {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
                }
                
            else:
                self.logger.error(f"Unbekannter Modelltyp: {model_type}")
                return None, {}
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Fehler beim Training des {model_type}-Modells: {str(e)}")
            return None, {}
    
    def _save_model(self, model_name: str, model: Any, model_type: str) -> bool:
        """
        Speichert ein trainiertes Modell.
        
        Args:
            model_name: Name des Modells
            model: Trainiertes Modell
            model_type: Typ des Modells
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Modell im Speicher cachen
            self.models[model_name] = model
            
            # Modell auf Festplatte speichern
            model_path = self.models_path / f"{model_name}_{model_type}.pkl"
            
            if model_type == 'lstm':
                # TensorFlow-Modell speichern
                model_path = self.models_path / model_name
                save_model(model, model_path)
            else:
                # Sklearn-Modell speichern
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            self.logger.info(f"Modell {model_name} gespeichert unter {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Modells {model_name}: {str(e)}")
            return False
    
    def load_model(self, model_name: str) -> Any:
        """
        Lädt ein trainiertes Modell.
        
        Args:
            model_name: Name des zu ladenden Modells
            
        Returns:
            Geladenes Modell oder None bei Fehler
        """
        # Prüfen, ob das Modell bereits im Speicher ist
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            # Versuche, das Modell von der Festplatte zu laden
            # Prüfe zunächst, ob es ein LSTM-Modell ist
            lstm_path = self.models_path / model_name
            if lstm_path.exists():
                model = load_model(lstm_path)
                self.models[model_name] = model
                return model
            
            # Suche nach Sklearn-Modellen
            for model_type in ['random_forest', 'gradient_boosting']:
                model_path = self.models_path / f"{model_name}_{model_type}.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_name] = model
                        return model
            
            self.logger.warning(f"Kein Modell mit Namen {model_name} gefunden")
            return None
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells {model_name}: {str(e)}")
            return None
    
    def predict(self, model_name: str, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Erstellt eine Vorhersage mit einem trainierten Modell.
        
        Args:
            model_name: Name des Modells
            data: Dictionary mit Feature-Werten
            
        Returns:
            Vorhersage-Ergebnis als Dictionary
        """
        try:
            model = self.load_model(model_name)
            
            if model is None:
                return {
                    'success': False,
                    'error': f"Modell {model_name} nicht gefunden"
                }
            
            # Performance-Metriken abrufen
            metrics = self.performance_metrics.get(model_name, {})
            features = metrics.get('features', self.default_features)
            model_type = metrics.get('model_type', 'unknown')
            
            # Prüfen, ob alle erforderlichen Features vorhanden sind
            missing_features = [f for f in features if f not in data]
            if missing_features:
                return {
                    'success': False,
                    'error': f"Fehlende Features: {missing_features}"
                }
            
            # Feature-Werte extrahieren und normalisieren
            X = np.array([[data[f] for f in features]])
            
            # Normalisierung (einfache Z-Score-Normalisierung, ohne zu trainieren)
            X = (X - np.mean(X)) / np.std(X)
            
            # Vorhersage basierend auf Modelltyp
            if model_type == 'lstm':
                X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
                prediction = model.predict(X_reshaped)[0][0]
                confidence = float(prediction)
                label = 1 if prediction > 0.5 else 0
            else:
                label = model.predict(X)[0]
                # Konfidenz aus Wahrscheinlichkeiten berechnen, falls verfügbar
                if hasattr(model, 'predict_proba'):
                    confidence = float(model.predict_proba(X)[0][label])
                else:
                    confidence = None
            
            return {
                'success': True,
                'model': model_name,
                'prediction': int(label),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Vorhersage mit Modell {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def backtest_model(self, model_name: str, symbol: Optional[str] = None, 
                      timeframe: Optional[str] = None, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Führt einen Backtest für ein trainiertes Modell durch.
        
        Args:
            model_name: Name des Modells
            symbol: Handelssymbol (optional, falls nicht im Modell gespeichert)
            timeframe: Zeitrahmen (optional, falls nicht im Modell gespeichert)
            start_date: Startdatum für den Backtest (optional)
            end_date: Enddatum für den Backtest (optional)
            
        Returns:
            Ergebnisse des Backtests als Dictionary
        """
        if self.is_backtesting:
            self.logger.warning(f"Backtest für {model_name} kann nicht gestartet werden, da bereits ein Backtest läuft")
            return {
                'success': False,
                'error': "Ein anderer Backtest läuft bereits"
            }
        
        if not self.data_pipeline:
            self.logger.error("Keine Datenpipeline verfügbar, Backtesting nicht möglich")
            return {
                'success': False,
                'error': "Keine Datenpipeline verfügbar"
            }
        
        # Modell laden oder Fehler zurückgeben
        model = self.load_model(model_name)
        
        if model is None:
            return {
                'success': False,
                'error': f"Modell {model_name} nicht gefunden"
            }
        
        # Modelldetails abrufen
        metrics = self.performance_metrics.get(model_name, {})
        model_symbol = symbol or metrics.get('symbol')
        model_timeframe = timeframe or metrics.get('timeframe')
        model_type = metrics.get('model_type', 'unknown')
        features = metrics.get('features', self.default_features)
        
        if not model_symbol or not model_timeframe:
            return {
                'success': False,
                'error': "Symbol und Timeframe müssen entweder im Modell gespeichert oder als Parameter übergeben werden"
            }
        
        # Zeitraum festlegen, falls nicht angegeben
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            # Start-Datum basierend auf Backtest-Tagen berechnen
            start_date = (datetime.now() - timedelta(days=self.backtest_days)).strftime('%Y-%m-%d')
        
        # Backtest in separatem Thread starten
        self.is_backtesting = True
        self.backtesting_thread = threading.Thread(
            target=self._backtest_model_thread,
            args=(model_name, model, model_symbol, model_timeframe, features, start_date, end_date, model_type),
            daemon=True
        )
        self.backtesting_thread.start()
        
        return {
            'success': True,
            'message': f"Backtest für Modell {model_name} gestartet",
            'symbol': model_symbol,
            'timeframe': model_timeframe,
            'period': f"{start_date} bis {end_date}"
        }
    
    def _backtest_model_thread(self, model_name: str, model: Any, symbol: str, 
                             timeframe: str, features: List[str], start_date: str,
                             end_date: str, model_type: str):
        """
        Thread-Funktion für das Backtesting eines Modells.
        
        Args:
            model_name: Name des Modells
            model: Trainiertes Modell
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            features: Liste von Features
            start_date: Startdatum
            end_date: Enddatum
            model_type: Typ des Modells
        """
        try:
            self.logger.info(f"Backtest für {model_name} gestartet (Symbol: {symbol}, Timeframe: {timeframe})")
            
            # Daten laden
            df = self._load_and_prepare_data(symbol, timeframe, start_date, end_date)
            
            if df is None or df.empty:
                self.logger.error(f"Keine Daten für {symbol} ({timeframe}) verfügbar")
                self.is_backtesting = False
                return
            
            # Features vorbereiten
            X, y = self._prepare_features_and_target(df, features, 'direction')
            
            if X is None or y is None:
                self.logger.error(f"Fehler bei der Feature-Vorbereitung für {symbol}")
                self.is_backtesting = False
                return
            
            # Vorhersagen erstellen
            if model_type == 'lstm':
                X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
                predictions_prob = model.predict(X_reshaped).flatten()
                predictions = (predictions_prob > 0.5).astype(int)
            else:
                predictions = model.predict(X)
            
            # Ergebnisse dem DataFrame hinzufügen
            df = df.iloc[:len(predictions)]  # Sicherstellen, dass die Längen übereinstimmen
            df['prediction'] = predictions
            
            # Trading-Logik simulieren
            df['position'] = 0
            df['entry_price'] = None
            df['exit_price'] = None
            df['trade_profit'] = 0.0
            
            position_open = False
            entry_price = 0.0
            entry_index = 0
            trade_records = []
            
            for i in range(1, len(df)):
                if not position_open and df['prediction'].iloc[i-1] == 1:
                    # Long-Position eröffnen
                    position_open = True
                    entry_price = df['close'].iloc[i]
                    entry_index = i
                    df.at[df.index[i], 'position'] = 1
                    df.at[df.index[i], 'entry_price'] = entry_price
                
                elif position_open:
                    # Position schließen, wenn Vorhersage auf 0 wechselt oder nach max. Haltezeit
                    max_hold_periods = 5  # Maximale Haltezeit
                    
                    if df['prediction'].iloc[i-1] == 0 or (i - entry_index) >= max_hold_periods:
                        exit_price = df['close'].iloc[i]
                        profit_pct = (exit_price / entry_price - 1) * 100
                        
                        df.at[df.index[i], 'position'] = 0
                        df.at[df.index[i], 'exit_price'] = exit_price
                        df.at[df.index[i], 'trade_profit'] = profit_pct
                        
                        # Trade aufzeichnen
                        trade = {
                            'entry_time': df.index[entry_index],
                            'exit_time': df.index[i],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct,
                            'hold_periods': i - entry_index
                        }
                        trade_records.append(trade)
                        
                        position_open = False
                    else:
                        df.at[df.index[i], 'position'] = 1
            
            # Abschließende offene Position schließen
            if position_open:
                exit_price = df['close'].iloc[-1]
                profit_pct = (exit_price / entry_price - 1) * 100
                
                df.at[df.index[-1], 'position'] = 0
                df.at[df.index[-1], 'exit_price'] = exit_price
                df.at[df.index[-1], 'trade_profit'] = profit_pct
                
                # Trade aufzeichnen
                trade = {
                    'entry_time': df.index[entry_index],
                    'exit_time': df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'hold_periods': len(df) - 1 - entry_index
                }
                trade_records.append(trade)
            
            # Performance-Metriken berechnen
            trades_df = pd.DataFrame(trade_records)
            
            backtest_results = {
                'model_name': model_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'period': f"{start_date} bis {end_date}",
                'total_trades': len(trade_records),
                'winning_trades': len(trades_df[trades_df['profit_pct'] > 0]) if not trades_df.empty else 0,
                'losing_trades': len(trades_df[trades_df['profit_pct'] <= 0]) if not trades_df.empty else 0,
                'win_rate': len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) if not trades_df.empty else 0,
                'avg_profit': trades_df['profit_pct'].mean() if not trades_df.empty else 0,
                'avg_win': trades_df[trades_df['profit_pct'] > 0]['profit_pct'].mean() if not trades_df.empty and len(trades_df[trades_df['profit_pct'] > 0]) > 0 else 0,
                'avg_loss': trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].mean() if not trades_df.empty and len(trades_df[trades_df['profit_pct'] <= 0]) > 0 else 0,
                'max_profit': trades_df['profit_pct'].max() if not trades_df.empty else 0,
                'max_loss': trades_df['profit_pct'].min() if not trades_df.empty else 0,
                'total_profit': trades_df['profit_pct'].sum() if not trades_df.empty else 0,
                'avg_hold_periods': trades_df['hold_periods'].mean() if not trades_df.empty else 0,
                'accuracy': accuracy_score(y, predictions),
                'timestamp': datetime.now().isoformat(),
                'trades': trade_records
            }
            
            # Backtest-Ergebnis speichern
            self._save_backtest_results(model_name, backtest_results)
            
            # Performance-Metriken aktualisieren
            if model_name in self.performance_metrics:
                self.performance_metrics[model_name]['backtest_results'] = backtest_results
            
            # Visualisierung erstellen
            self._create_backtest_visualization(model_name, df, backtest_results)
            
            self.logger.info(f"Backtest für {model_name} abgeschlossen. "
                           f"Win Rate: {backtest_results['win_rate']:.2%}, "
                           f"Total Profit: {backtest_results['total_profit']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Fehler im Backtest Thread für {model_name}: {str(e)}")
        
        finally:
            self.is_backtesting = False
    
    def _save_backtest_results(self, model_name: str, results: Dict[str, Any]):
        """
        Speichert die Backtest-Ergebnisse.
        
        Args:
            model_name: Name des Modells
            results: Backtest-Ergebnisse
        """
        try:
            # Ergebnisse als JSON speichern
            results_path = self.results_path / f"{model_name}_backtest.json"
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # default=str für Datetime-Objekte
            
            self.logger.info(f"Backtest-Ergebnisse für {model_name} gespeichert unter {results_path}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Backtest-Ergebnisse für {model_name}: {str(e)}")
    
    def _create_backtest_visualization(self, model_name: str, df: pd.DataFrame, results: Dict[str, Any]):
        """
        Erstellt eine Visualisierung der Backtest-Ergebnisse.
        
        Args:
            model_name: Name des Modells
            df: DataFrame mit Handelsdaten und Ergebnissen
            results: Backtest-Ergebnisse
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Preischart mit Einstiegs- und Ausstiegspunkten
            ax1.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
            
            # Einstiegspunkte markieren
            entries = df[df['entry_price'].notnull()]
            ax1.scatter(entries.index, entries['entry_price'], marker='^', color='green', s=100, label='Entry')
            
            # Ausstiegspunkte markieren
            exits = df[df['exit_price'].notnull()]
            ax1.scatter(exits.index, exits['exit_price'], marker='v', color='red', s=100, label='Exit')
            
            ax1.set_title(f'Backtest Results for {model_name} ({results["symbol"]}, {results["timeframe"]})')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gewinn/Verlust pro Trade
            trade_profits = [trade['profit_pct'] for trade in results['trades']]
            trade_times = [trade['exit_time'] for trade in results['trades']]
            
            colors = ['green' if profit > 0 else 'red' for profit in trade_profits]
            ax2.bar(range(len(trade_profits)), trade_profits, color=colors)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Profit/Loss (%)')
            ax2.set_title(f'Trade Performance (Win Rate: {results["win_rate"]:.2%}, Total Profit: {results["total_profit"]:.2f}%)')
            ax2.grid(True, alpha=0.3)
            
            # Textbox mit Statistiken
            stats_text = (
                f"Total Trades: {results['total_trades']}\n"
                f"Winning Trades: {results['winning_trades']}\n"
                f"Losing Trades: {results['losing_trades']}\n"
                f"Win Rate: {results['win_rate']:.2%}\n"
                f"Average Profit: {results['avg_profit']:.2f}%\n"
                f"Average Win: {results['avg_win']:.2f}%\n"
                f"Average Loss: {results['avg_loss']:.2f}%\n"
                f"Maximum Profit: {results['max_profit']:.2f}%\n"
                f"Maximum Loss: {results['max_loss']:.2f}%"
            )
            
            # Textbox platzieren
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=props)
            
            plt.tight_layout()
            
            # Speichern der Visualisierung
            fig_path = self.results_path / f"{model_name}_backtest_visualization.png"
            plt.savefig(fig_path)
            plt.close(fig)
            
            self.logger.info(f"Backtest-Visualisierung für {model_name} gespeichert unter {fig_path}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Backtest-Visualisierung für {model_name}: {str(e)}")
    
    def get_market_signal(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Generiert ein Handelssignal für ein Symbol basierend auf allen trainierten Modellen.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            
        Returns:
            Handelssignal als Dictionary
        """
        try:
            if not self.data_pipeline:
                return {
                    'success': False,
                    'error': "Keine Datenpipeline verfügbar"
                }
            
            # Aktuelle Marktdaten abrufen
            df = self.data_pipeline.get_crypto_data(symbol, timeframe, limit=50)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': f"Keine Daten für {symbol} ({timeframe}) verfügbar"
                }
            
            # Technische Indikatoren hinzufügen
            indicators = [
                {'type': 'rsi', 'period': 14},
                {'type': 'ema', 'period': 9},
                {'type': 'ema', 'period': 21},
                {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                {'type': 'bollinger', 'period': 20, 'std_dev': 2},
                {'type': 'atr', 'period': 14}
            ]
            
            df = self.data_pipeline.get_indicators(df, indicators)
            
            # Neueste Daten extrahieren
            latest_data = df.iloc[-1].to_dict()
            
            # Alle passenden Modelle finden
            matching_models = []
            for model_name, metrics in self.performance_metrics.items():
                if metrics.get('symbol') == symbol and metrics.get('timeframe') == timeframe:
                    if 'backtest_results' in metrics and metrics['backtest_results'].get('win_rate', 0) >= 0.5:
                        matching_models.append(model_name)
            
            if not matching_models:
                return {
                    'success': False,
                    'error': f"Keine passenden Modelle für {symbol} ({timeframe}) gefunden"
                }
            
            # Vorhersagen von allen Modellen sammeln
            predictions = []
            for model_name in matching_models:
                prediction = self.predict(model_name, latest_data)
                if prediction['success']:
                    predictions.append({
                        'model': model_name,
                        'prediction': prediction['prediction'],
                        'confidence': prediction.get('confidence', 0.5)
                    })
            
            if not predictions:
                return {
                    'success': False,
                    'error': "Keine erfolgreichen Vorhersagen"
                }
            
            # Gesamtsignal basierend auf einem gewichteten Durchschnitt berechnen
            total_confidence = 0
            weighted_sum = 0
            
            for pred in predictions:
                confidence = pred.get('confidence', 0.5)
                total_confidence += confidence
                weighted_sum += pred['prediction'] * confidence
            
            if total_confidence > 0:
                signal = weighted_sum / total_confidence
            else:
                signal = 0.5
            
            # Signal-Richtung bestimmen
            direction = "buy" if signal > 0.6 else "sell" if signal < 0.4 else "neutral"
            
            # Aktuellen Preis und technische Indikatoren hinzufügen
            current_price = latest_data['close']
            rsi = latest_data.get('rsi_14', None)
            
            # Traditionelle Signale zur Bestätigung
            traditional_signals = []
            
            if rsi is not None:
                if rsi < 30:
                    traditional_signals.append("RSI oversold")
                elif rsi > 70:
                    traditional_signals.append("RSI overbought")
            
            if 'ema_9' in latest_data and 'ema_21' in latest_data:
                ema_9 = latest_data['ema_9']
                ema_21 = latest_data['ema_21']
                
                if ema_9 > ema_21:
                    traditional_signals.append("EMA9 above EMA21 (bullish)")
                else:
                    traditional_signals.append("EMA9 below EMA21 (bearish)")
            
            # Signalstärke basierend auf Übereinstimmung berechnen
            if direction == "buy":
                signal_strength = min(1.0, signal * 1.5)  # Skalieren auf max. 1.0
            elif direction == "sell":
                signal_strength = min(1.0, (1 - signal) * 1.5)  # Skalieren auf max. 1.0
            else:
                signal_strength = 0.5  # Neutral = 50% Stärke
            
            # Ergebnis zusammenstellen
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'direction': direction,
                'signal_value': signal,
                'signal_strength': signal_strength,
                'rsi': rsi,
                'traditional_signals': traditional_signals,
                'model_predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Signal im Cache speichern
            self.current_signals[f"{symbol}_{timeframe}"] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Generierung des Handelssignals für {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Führt eine umfassende Analyse eines Symbols durch.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            
        Returns:
            Analyseergebnis als Dictionary
        """
        try:
            if not self.data_pipeline:
                return {
                    'success': False,
                    'error': "Keine Datenpipeline verfügbar"
                }
            
            # Aktuelle Marktdaten abrufen
            df = self.data_pipeline.get_crypto_data(symbol, timeframe, limit=100)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': f"Keine Daten für {symbol} ({timeframe}) verfügbar"
                }
            
            # Technische Indikatoren hinzufügen
            indicators = [
                {'type': 'rsi', 'period': 14},
                {'type': 'ema', 'period': 9},
                {'type': 'ema', 'period': 21},
                {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                {'type': 'bollinger', 'period': 20, 'std_dev': 2},
                {'type': 'atr', 'period': 14}
            ]
            
            df = self.data_pipeline.get_indicators(df, indicators)
            
            # Zukünftige Preisänderungen berechnen (für die Zielwerte)
            df['next_close'] = df['close'].shift(-1)
            df['price_change'] = df['next_close'] - df['close']
            df['returns'] = df['price_change'] / df['close']
            df['direction'] = (df['returns'] > 0).astype(int)
            
            # Aktuellen Preisstand und Änderung berechnen
            current_price = df['close'].iloc[-1]
            price_change_1d = (current_price / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0
            price_change_7d = (current_price / df['close'].iloc[-7] - 1) * 100 if len(df) > 6 else 0
            
            # Technische Analysen
            rsi = df['rsi_14'].iloc[-1]
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_cross = "bullish" if ema_9 > ema_21 else "bearish"
            
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            macd_cross = "bullish" if macd > macd_signal else "bearish"
            
            bband_upper = df['bband_upper_20'].iloc[-1]
            bband_lower = df['bband_lower_20'].iloc[-1]
            bband_position = (current_price - bband_lower) / (bband_upper - bband_lower) if bband_upper != bband_lower else 0.5
            
            atr = df['atr_14'].iloc[-1]
            volatility = df['close'].pct_change().rolling(window=14).std().iloc[-1] * 100
            
            # Signalgenerator aufrufen
            trading_signal = self.get_market_signal(symbol, timeframe)
            
            # Traditionelle Signale zur Bestätigung
            signals = []
            
            if rsi < 30:
                signals.append({"indicator": "RSI", "signal": "buy", "strength": 0.7, "description": "Oversold"})
            elif rsi > 70:
                signals.append({"indicator": "RSI", "signal": "sell", "strength": 0.7, "description": "Overbought"})
            
            if ema_9 > ema_21:
                signals.append({"indicator": "EMA Cross", "signal": "buy", "strength": 0.6, "description": "Short-term EMA above long-term EMA"})
            else:
                signals.append({"indicator": "EMA Cross", "signal": "sell", "strength": 0.6, "description": "Short-term EMA below long-term EMA"})
            
            if macd > macd_signal:
                signals.append({"indicator": "MACD", "signal": "buy", "strength": 0.65, "description": "MACD above signal line"})
            else:
                signals.append({"indicator": "MACD", "signal": "sell", "strength": 0.65, "description": "MACD below signal line"})
            
            if current_price < bband_lower:
                signals.append({"indicator": "Bollinger Bands", "signal": "buy", "strength": 0.7, "description": "Price below lower band"})
            elif current_price > bband_upper:
                signals.append({"indicator": "Bollinger Bands", "signal": "sell", "strength": 0.7, "description": "Price above upper band"})
            
            # Support- und Widerstandsniveaus bestimmen
            price_history = df['close'].tolist()
            support_levels = self._find_support_levels(price_history, current_price)
            resistance_levels = self._find_resistance_levels(price_history, current_price)
            
            # Trendanalyse
            trend_30d = self._analyze_trend(df, days=30)
            trend_7d = self._analyze_trend(df, days=7)
            
            # Gesamtanalyse erstellen
            analysis = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'price_change_1d': price_change_1d,
                'price_change_7d': price_change_7d,
                'technical_indicators': {
                    'rsi': rsi,
                    'ema_cross': ema_cross,
                    'macd_cross': macd_cross,
                    'bollinger_band_position': bband_position,
                    'atr': atr,
                    'volatility': volatility
                },
                'signals': signals,
                'trend': {
                    'long_term': trend_30d,
                    'short_term': trend_7d
                },
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'model_signal': trading_signal if trading_signal.get('success', False) else None,
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Analyse von {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _find_support_levels(self, prices: List[float], current_price: float, max_levels: int = 3) -> List[float]:
        """
        Findet Support-Levels unterhalb des aktuellen Preises.
        
        Args:
            prices: Liste von historischen Preisen
            current_price: Aktueller Preis
            max_levels: Maximale Anzahl an zurückzugebenden Levels
            
        Returns:
            Liste von Support-Levels
        """
        support_levels = []
        price_array = np.array(prices)
        
        # Lokale Minima finden
        for i in range(2, len(price_array) - 2):
            if (price_array[i-2] > price_array[i-1] > price_array[i] < price_array[i+1] < price_array[i+2] and
                price_array[i] < current_price):
                support_levels.append(price_array[i])
        
        # Duplikate entfernen und sortieren
        support_levels = sorted(set(support_levels), reverse=True)
        
        # Auf max_levels begrenzen
        return support_levels[:max_levels]
    
    def _find_resistance_levels(self, prices: List[float], current_price: float, max_levels: int = 3) -> List[float]:
        """
        Findet Widerstands-Levels oberhalb des aktuellen Preises.
        
        Args:
            prices: Liste von historischen Preisen
            current_price: Aktueller Preis
            max_levels: Maximale Anzahl an zurückzugebenden Levels
            
        Returns:
            Liste von Widerstands-Levels
        """
        resistance_levels = []
        price_array = np.array(prices)
        
        # Lokale Maxima finden
        for i in range(2, len(price_array) - 2):
            if (price_array[i-2] < price_array[i-1] < price_array[i] > price_array[i+1] > price_array[i+2] and
                price_array[i] > current_price):
                resistance_levels.append(price_array[i])
        
        # Duplikate entfernen und sortieren
        resistance_levels = sorted(set(resistance_levels))
        
        # Auf max_levels begrenzen
        return resistance_levels[:max_levels]
    
    def _analyze_trend(self, df: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
        """
        Analysiert den Trend der letzten X Tage.
        
        Args:
            df: DataFrame mit Preisdaten
            days: Anzahl der Tage für die Trendanalyse
            
        Returns:
            Trendanalyse als Dictionary
        """
        if len(df) < days:
            days = len(df)
        
        # Daten für die Trendanalyse extrahieren
        trend_data = df.iloc[-days:]
        
        # Lineare Regression für Trendlinie
        x = np.arange(len(trend_data))
        y = trend_data['close'].values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Bestimme Trend-Richtung und -Stärke
        if slope > 0:
            direction = "bullish"
            strength = min(1.0, slope * 100 / y.mean())  # Normalisieren
        else:
            direction = "bearish"
            strength = min(1.0, abs(slope) * 100 / y.mean())  # Normalisieren
        
        # Volatilität und Konsistenz des Trends
        price_changes = trend_data['close'].pct_change().dropna()
        volatility = price_changes.std() * 100
        
        # Zähle wie oft die Preise die Trendlinie kreuzen (weniger = konsistenter Trend)
        trend_line = intercept + slope * x
        crosses = sum(1 for i in range(1, len(y)) if (y[i-1] - trend_line[i-1]) * (y[i] - trend_line[i]) < 0)
        consistency = 1 - (crosses / len(y))
        
        # Qualitative Bewertung
        if strength < 0.2:
            description = "Very weak"
        elif strength < 0.4:
            description = "Weak"
        elif strength < 0.6:
            description = "Moderate"
        elif strength < 0.8:
            description = "Strong"
        else:
            description = "Very strong"
        
        description = f"{description} {direction} trend"
        
        return {
            'direction': direction,
            'strength': strength,
            'volatility': volatility,
            'consistency': consistency,
            'description': description,
            'slope': slope,
            'intercept': intercept
        }
    
    def get_active_models(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste der aktiven Modelle zurück, optional gefiltert nach Symbol.
        
        Args:
            symbol: Optionales Handelssymbol zur Filterung
            
        Returns:
            Liste von Modell-Informationen
        """
        active_models = []
        
        for model_name, metrics in self.performance_metrics.items():
            model_symbol = metrics.get('symbol')
            
            if symbol is None or model_symbol == symbol:
                model_info = {
                    'name': model_name,
                    'symbol': model_symbol,
                    'timeframe': metrics.get('timeframe'),
                    'type': metrics.get('model_type'),
                    'last_updated': metrics.get('last_updated'),
                    'accuracy': metrics.get('train_metrics', {}).get('accuracy')
                }
                
                # Backtest-Ergebnisse hinzufügen, falls vorhanden
                if 'backtest_results' in metrics:
                    backtest = metrics['backtest_results']
                    model_info.update({
                        'win_rate': backtest.get('win_rate'),
                        'total_trades': backtest.get('total_trades'),
                        'total_profit': backtest.get('total_profit')
                    })
                
                active_models.append(model_info)
        
        return active_models
    
    def get_backtest_results(self, model_name: str) -> Dict[str, Any]:
        """
        Gibt die Backtest-Ergebnisse für ein bestimmtes Modell zurück.
        
        Args:
            model_name: Name des Modells
            
        Returns:
            Backtest-Ergebnisse als Dictionary oder Fehlermeldung
        """
        try:
            # Prüfen, ob Backtest-Ergebnisse im Speicher vorhanden sind
            if model_name in self.performance_metrics and 'backtest_results' in self.performance_metrics[model_name]:
                return {
                    'success': True,
                    'results': self.performance_metrics[model_name]['backtest_results']
                }
            
            # Versuche, die Ergebnisse von der Festplatte zu laden
            results_path = self.results_path / f"{model_name}_backtest.json"
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                return {
                    'success': True,
                    'results': results
                }
            
            return {
                'success': False,
                'error': f"Keine Backtest-Ergebnisse für Modell {model_name} gefunden"
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Backtest-Ergebnisse für {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Erstellt eine Zusammenfassung für ein bestimmtes Modell.
        
        Args:
            model_name: Name des Modells
            
        Returns:
            Modell-Zusammenfassung als Dictionary
        """
        try:
            if model_name not in self.performance_metrics:
                return {
                    'success': False,
                    'error': f"Modell {model_name} nicht gefunden"
                }
            
            metrics = self.performance_metrics[model_name]
            
            # Modell-Details sammeln
            model_summary = {
                'name': model_name,
                'symbol': metrics.get('symbol'),
                'timeframe': metrics.get('timeframe'),
                'model_type': metrics.get('model_type'),
                'training_date': metrics.get('last_updated'),
                'features': metrics.get('features'),
                'training_metrics': metrics.get('train_metrics', {})
            }
            
            # Backtest-Ergebnisse, falls vorhanden
            if 'backtest_results' in metrics:
                backtest = metrics['backtest_results']
                model_summary['backtest'] = {
                    'win_rate': backtest.get('win_rate'),
                    'total_trades': backtest.get('total_trades'),
                    'total_profit': backtest.get('total_profit'),
                    'avg_profit': backtest.get('avg_profit'),
                    'avg_win': backtest.get('avg_win'),
                    'avg_loss': backtest.get('avg_loss'),
                    'max_profit': backtest.get('max_profit'),
                    'max_loss': backtest.get('max_loss')
                }
            
            # Feature-Importance, falls vorhanden
            if 'train_metrics' in metrics and 'feature_importance' in metrics['train_metrics']:
                # Feature-Namen und Importance-Werte kombinieren
                features = metrics.get('features', [])
                importance = metrics['train_metrics']['feature_importance']
                
                if len(features) == len(importance):
                    feature_importance = dict(zip(features, importance))
                    # Nach Wichtigkeit sortieren
                    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                    model_summary['feature_importance'] = feature_importance
            
            return {
                'success': True,
                'summary': model_summary
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen der Modell-Zusammenfassung für {model_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Status des Learning-Moduls zurück.
        
        Returns:
            Status als Dictionary
        """
        return {
            'is_training': self.is_training,
            'is_backtesting': self.is_backtesting,
            'active_models': len(self.models),
            'model_metrics': len(self.performance_metrics),
            'active_signals': len(self.current_signals),
            'last_updated': datetime.now().isoformat()
        }

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'backtest_days': 90,
        'paper_trading_days': 14,
        'target_win_rate': 0.6,
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'patience': 10
        }
    }
    
    # Learning-Modul initialisieren
    learning_module = LearningModule(config)
    
    # Hier würde man die Datenpipeline setzen
    # learning_module.set_data_pipeline(data_pipeline)
    
    # Modell trainieren
    # learning_module.train_model('btc_direction_1h', 'BTC/USDT', '1h', model_type='random_forest')
    
    # Handelssignal abrufen
    # signal = learning_module.get_market_signal('BTC/USDT', '1h')
    # print(f"Signal: {signal}")

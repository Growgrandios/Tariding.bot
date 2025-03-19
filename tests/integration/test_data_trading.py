import unittest
import os
import pandas as pd
from src.core.config_manager import ConfigManager
from src.modules.data_pipeline import DataPipeline
from src.modules.live_trading import LiveTradingConnector

class TestDataTradingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Einmalige Initialisierung für alle Tests"""
        # Konfigurations-Dummy erstellen
        cls.config = {
            "data_pipeline": {
                "update_intervals": {"crypto": 60},
                "crypto_assets": ["BTC/USDT:USDT"],
                "timeframes": ["1h"]
            },
            "trading": {
                "mode": "paper",
                "risk_per_trade": 0.01
            }
        }
        
        # Testdaten erstellen
        cls.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        
    def test_data_to_trading_flow(self):
        """Testet den Datenfluss von DataPipeline zu LiveTrading"""
        # Mocks erstellen und injizieren
        # In einem echten Test würden Sie Mock-Objekte verwenden
        
        # Hier ist ein Beispiel für einen vereinfachten Test:
        data_pipeline = DataPipeline(self.config['data_pipeline'])
        trading_connector = LiveTradingConnector(self.config['trading'])
        
        # DataPipeline mit Testdaten füllen
        # In einem echten Test würden Sie die APIs mocken
        data_pipeline.test_inject_data("BTC/USDT:USDT", "1h", self.test_data)
        
        # Handelslogik mit Daten aus der Pipeline ausführen
        latest_data = data_pipeline.get_latest_data("BTC/USDT:USDT", "1h", 10)
        result = trading_connector.analyze_market(latest_data)
        
        # Überprüfen, ob Daten korrekt fließen
        self.assertIsNotNone(result)
        self.assertTrue('signal' in result)

if __name__ == '__main__':
    unittest.main()

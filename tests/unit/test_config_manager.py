import unittest
import os
import tempfile
import yaml
from src.core.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Temporäre Testdatei erstellen
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_dir = os.path.join(self.test_dir.name, "config")
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.config_file = os.path.join(self.config_dir, "test_config.yaml")
        self.test_config = {
            "general": {
                "bot_name": "TestBot",
                "log_level": "INFO"
            },
            "trading": {
                "mode": "paper",
                "risk_per_trade": 0.01
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
            
        self.config_manager = ConfigManager(self.config_file)
    
    def tearDown(self):
        # Temporäres Verzeichnis aufräumen
        self.test_dir.cleanup()
    
    def test_load_config(self):
        """Test, ob Konfiguration korrekt geladen wird"""
        config = self.config_manager.get_config()
        self.assertEqual(config["general"]["bot_name"], "TestBot")
        self.assertEqual(config["trading"]["mode"], "paper")
    
    def test_update_config(self):
        """Test, ob Konfiguration korrekt aktualisiert wird"""
        # Konfiguration ändern
        self.config_manager.update_section("trading", {"mode": "live"})
        
        # Überprüfen, ob Änderung übernommen wurde
        config = self.config_manager.get_config()
        self.assertEqual(config["trading"]["mode"], "live")
        
        # Überprüfen, ob Änderung gespeichert wurde
        with open(self.config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        self.assertEqual(saved_config["trading"]["mode"], "live")

if __name__ == '__main__':
    unittest.main()

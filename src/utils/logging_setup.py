import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
import json

class DynamicLogLevelFilter(logging.Filter):
    """Filter zum dynamischen Anpassen des Log-Levels zur Laufzeit"""
    
    def __init__(self, default_level=logging.INFO):
        super().__init__()
        self.min_level = default_level
    
    def filter(self, record):
        return record.levelno >= self.min_level

class LogManager:
    """Manager für zentrales Logging mit dynamischer Anpassung"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.loggers = {}
            cls._instance.log_filters = {}
        return cls._instance
    
    def setup_logging(self, log_dir="logs", log_level=logging.INFO, 
                      rotation_type="size", backup_count=5,
                      log_config_file="data/config/logging_config.json"):
        """Richtet zentrales Logging mit konfigurierbarer Rotation ein."""
        if self.initialized:
            return
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Standardkonfiguration
        self.log_config = {
            "global_level": log_level,
            "console_level": log_level,
            "file_level": log_level,
            "rotation_type": rotation_type,  # "size" oder "time"
            "rotation_size_mb": 10,          # für size-basierte Rotation
            "rotation_interval": "midnight",  # für zeitbasierte Rotation
            "backup_count": backup_count,
            "module_levels": {
                "core": logging.INFO,
                "data": logging.INFO,
                "trading": logging.INFO,
                "learning": logging.INFO,
                "black_swan": logging.INFO,
                "telegram": logging.INFO,
                "tax": logging.INFO,
                "security": logging.INFO,
                "backup": logging.INFO,
                "credentials": logging.INFO
            }
        }
        
        # Lade Konfiguration aus Datei, falls vorhanden
        if os.path.exists(log_config_file):
            try:
                with open(log_config_file, 'r') as f:
                    user_config = json.load(f)
                    # Rekursives Update der Konfiguration
                    self._update_config(self.log_config, user_config)
            except Exception as e:
                print(f"Fehler beim Laden der Logging-Konfiguration: {str(e)}")
        else:
            # Speichere Standardkonfiguration
            os.makedirs(os.path.dirname(log_config_file), exist_ok=True)
            with open(log_config_file, 'w') as f:
                json.dump(self.log_config, f, indent=4)
        
        # Root-Logger konfigurieren
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_config["global_level"])
        
        # Alle vorhandenen Handler entfernen
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Formatierung erstellen
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Konsolenausgabe
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_filter = DynamicLogLevelFilter(self.log_config["console_level"])
        console_handler.addFilter(console_filter)
        root_logger.addHandler(console_handler)
        self.log_filters["console"] = console_filter
        
        # Dateihandler mit konfigurierbarer Rotation
        log_file = os.path.join(log_dir, "trading_bot.log")
        
        if self.log_config["rotation_type"] == "size":
            # Größenbasierte Rotation
            max_bytes = self.log_config["rotation_size_mb"] * 1024 * 1024
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=self.log_config["backup_count"]
            )
        else:
            # Zeitbasierte Rotation
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=self.log_config["rotation_interval"],
                backupCount=self.log_config["backup_count"]
            )
        
        file_handler.setFormatter(formatter)
        file_filter = DynamicLogLevelFilter(self.log_config["file_level"])
        file_handler.addFilter(file_filter)
        root_logger.addHandler(file_handler)
        self.log_filters["file"] = file_filter
        
        # Modulspezifische Logger konfigurieren
        modules = list(self.log_config["module_levels"].keys())
        
        for module in modules:
            module_log_file = os.path.join(log_dir, f"{module}.log")
            module_logger = logging.getLogger(module)
            
            # Alle vorhandenen Handler entfernen
            for handler in module_logger.handlers[:]:
                module_logger.removeHandler(handler)
            
            if self.log_config["rotation_type"] == "size":
                module_handler = RotatingFileHandler(
                    module_log_file,
                    maxBytes=5*1024*1024,  # 5 MB
                    backupCount=3
                )
            else:
                module_handler = TimedRotatingFileHandler(
                    module_log_file,
                    when="midnight",
                    backupCount=3
                )
            
            module_handler.setFormatter(formatter)
            module_level = self.log_config["module_levels"].get(module, logging.INFO)
            module_filter = DynamicLogLevelFilter(module_level)
            module_handler.addFilter(module_filter)
            module_logger.addHandler(module_handler)
            self.log_filters[module] = module_filter
            self.loggers[module] = module_logger
        
        self.initialized = True
        logging.info("Logging-System initialisiert")
    
    def _update_config(self, target, source):
        """Rekursives Update der Konfiguration"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def set_log_level(self, logger_name=None, level=None):
        """Ändert das Log-Level eines Loggers zur Laufzeit"""
        if not self.initialized:
            return False
        
        if level is None:
            return False
        
        if logger_name is None or logger_name == "global":
            # Globales Log-Level ändern
            self.log_config["global_level"] = level
            logging.getLogger().setLevel(level)
            return True
        
        if logger_name == "console":
            # Konsolen-Log-Level ändern
            self.log_config["console_level"] = level
            if "console" in self.log_filters:
                self.log_filters["console"].min_level = level
            return True
        
        if logger_name == "file":
            # Datei-Log-Level ändern
            self.log_config["file_level"] = level
            if "file" in self.log_filters:
                self.log_filters["file"].min_level = level
            return True
        
        # Modulspezifisches Log-Level ändern
        if logger_name in self.log_config["module_levels"]:
            self.log_config["module_levels"][logger_name] = level
            if logger_name in self.log_filters:
                self.log_filters[logger_name].min_level = level
            return True
        
        return False
    
    def save_config(self, config_file="data/config/logging_config.json"):
        """Speichert die aktuelle Logging-Konfiguration"""
        if not self.initialized:
            return False
        
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.log_config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Logging-Konfiguration: {str(e)}")
            return False

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Wrapper für die Kompatibilität mit bestehendem Code"""
    log_manager = LogManager()
    log_manager.setup_logging(log_dir=log_dir, log_level=log_level)
    return logging.getLogger()

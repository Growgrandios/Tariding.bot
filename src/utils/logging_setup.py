# logging_setup.py

import os
import logging
import json
import sys
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

class DynamicLogLevelFilter(logging.Filter):
    """
    Filter zum dynamischen Anpassen des Log-Levels zur Laufzeit.
    Ermöglicht die Änderung von Log-Levels ohne Neustart der Anwendung.
    """
    def __init__(self, default_level=logging.INFO):
        """
        Initialisiert den Filter mit einem Standard-Log-Level.
        
        Args:
            default_level: Standard-Log-Level (default: logging.INFO)
        """
        super().__init__()
        self.min_level = default_level

    def filter(self, record):
        """
        Filtert Log-Einträge basierend auf dem aktuellen dynamischen Level.
        
        Args:
            record: Der Log-Eintrag
            
        Returns:
            True, wenn der Eintrag das Mindestlevel erreicht oder überschreitet
        """
        return record.levelno >= self.min_level

class LogManager:
    """
    Manager für zentrales Logging mit dynamischer Anpassung der Log-Level.
    Implementiert das Singleton-Pattern für eine zentrale Logging-Instanz.
    """
    _instance = None

    def __new__(cls):
        """
        Implementiert das Singleton-Pattern für LogManager.
        
        Returns:
            Die einzige Instanz des LogManager
        """
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.loggers = {}
            cls._instance.log_filters = {}
            cls._instance.log_config = {}
        return cls._instance

    def setup_logging(self, 
                     log_dir: str = "logs", 
                     log_level: int = logging.INFO,
                     rotation_type: str = "size", 
                     backup_count: int = 5,
                     log_config_file: str = "data/config/logging_config.json",
                     component_name: Optional[str] = None) -> logging.Logger:
        """
        Richtet zentrales Logging mit konfigurierbarer Rotation ein.
        
        Args:
            log_dir: Verzeichnis für Log-Dateien
            log_level: Standard-Log-Level
            rotation_type: Art der Log-Rotation ('size' oder 'time')
            backup_count: Anzahl der zu behaltenden Backup-Dateien
            log_config_file: Pfad zur Logging-Konfigurationsdatei
            component_name: Name der Komponente (wird als Logger-Name verwendet)
            
        Returns:
            Konfigurierter Logger für die angegebene Komponente
        """
        # Verzeichnis erstellen, falls es nicht existiert
        os.makedirs(log_dir, exist_ok=True)
        
        # Verwende component_name für logger_name, falls angegeben
        logger_name = component_name or "gemma_bot"
        
        # Wenn bereits initialisiert, gib den vorhandenen Logger zurück
        if self.initialized and logger_name in self.loggers:
            return self.loggers[logger_name]
        
        # Erstelle einen Logger für die Komponente
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Lösche bestehende Handler, um Duplikate zu vermeiden
        logger.handlers = []
        
        # Formatter für konsistentes Log-Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        if rotation_type == "size":
            file_handler = RotatingFileHandler(
                f"{log_dir}/{logger_name}.log",
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=backup_count
            )
        else:  # time-based rotation
            file_handler = TimedRotatingFileHandler(
                f"{log_dir}/{logger_name}.log",
                when='midnight',
                interval=1,
                backupCount=backup_count
            )
            
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Filter für dynamisches Log-Level
        log_filter = DynamicLogLevelFilter(log_level)
        logger.addFilter(log_filter)
        
        # Speichere Filter für späteren Zugriff
        self.log_filters[logger_name] = log_filter
        
        # Speichere Logger für späteren Zugriff
        self.loggers[logger_name] = logger
        
        # Standard-Konfiguration
        self.log_config = {
            "global_level": log_level,
            "console_level": log_level,
            "file_level": log_level,
            "rotation_type": rotation_type,  # "size" oder "time"
            "rotation_size_mb": 10,  # für size-basierte Rotation
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
                logger.error(f"Fehler beim Laden der Logging-Konfiguration: {str(e)}")
        else:
            # Speichere Standardkonfiguration
            os.makedirs(os.path.dirname(log_config_file), exist_ok=True)
            try:
                with open(log_config_file, 'w') as f:
                    json.dump(self.log_config, f, indent=4)
            except Exception as e:
                logger.error(f"Fehler beim Speichern der Standard-Logging-Konfiguration: {str(e)}")
        
        self.initialized = True
        logger.info(f"Logging für {logger_name} erfolgreich initialisiert")
        
        return logger

    def setup_module_logging(self) -> None:
        """
        Richtet modulspezifische Logger basierend auf der Konfiguration ein.
        """
        if not hasattr(self, 'log_config') or not self.log_config:
            print("Keine Logging-Konfiguration vorhanden. Bitte zuerst setup_logging() aufrufen.")
            return
            
        log_dir = "logs"  # Standard-Verzeichnis
        
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
                    when=self.log_config["rotation_interval"],
                    backupCount=3
                )
                
            module_handler.setFormatter(formatter)
            module_filter = DynamicLogLevelFilter(self.log_config["module_levels"][module])
            module_handler.addFilter(module_filter)
            module_logger.addHandler(module_handler)
            self.log_filters[module] = module_filter
            
            # Speichere Logger für späteren Zugriff
            self.loggers[module] = module_logger

    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Rekursives Update der Konfiguration.
        
        Args:
            target: Ziel-Dictionary
            source: Quell-Dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value

    def set_log_level(self, 
                      logger_name: Optional[str] = None, 
                      level: Optional[int] = None) -> bool:
        """
        Ändert das Log-Level eines Loggers zur Laufzeit.
        
        Args:
            logger_name: Name des Loggers ('global', 'console', 'file' oder Modulname)
            level: Neues Log-Level
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.initialized:
            print("LogManager nicht initialisiert. Bitte zuerst setup_logging() aufrufen.")
            return False
            
        if level is None:
            return False
            
        try:
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
                
            # Spezifischen Logger ändern
            if logger_name in self.loggers:
                if logger_name in self.log_filters:
                    self.log_filters[logger_name].min_level = level
                return True
                
            return False
        except Exception as e:
            print(f"Fehler beim Ändern des Log-Levels: {str(e)}")
            return False

    def save_config(self, config_file: str = "data/config/logging_config.json") -> bool:
        """
        Speichert die aktuelle Logging-Konfiguration.
        
        Args:
            config_file: Pfad zur Konfigurationsdatei
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if not self.initialized:
            print("LogManager nicht initialisiert. Bitte zuerst setup_logging() aufrufen.")
            return False
            
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.log_config, f, indent=4)
            return True
        except Exception as e:
            print(f"Fehler beim Speichern der Logging-Konfiguration: {str(e)}")
            logging.error(f"Fehler beim Speichern der Logging-Konfiguration: {str(e)}")
            return False
            
    def get_logger(self, name: str) -> logging.Logger:
        """
        Gibt einen Logger mit dem angegebenen Namen zurück.
        Erstellt einen neuen Logger, falls keiner existiert.
        
        Args:
            name: Name des Loggers
            
        Returns:
            Logger-Instanz
        """
        if name in self.loggers:
            return self.loggers[name]
            
        # Erstelle einen neuen Logger und verwende Standardeinstellungen
        return self.setup_logging(component_name=name)

# Globale Hilfsfunktionen für einfachen Zugriff

def setup_logging(log_dir: str = "logs", 
                  log_level: int = logging.INFO,
                  component_name: Optional[str] = None) -> logging.Logger:
    """
    Wrapper für die Kompatibilität mit bestehendem Code.
    Erstellt und konfiguriert einen Logger.
    
    Args:
        log_dir: Verzeichnis für Log-Dateien
        log_level: Standard-Log-Level
        component_name: Name der Komponente
        
    Returns:
        Konfigurierter Logger
    """
    log_manager = LogManager()
    return log_manager.setup_logging(
        log_dir=log_dir,
        log_level=log_level,
        component_name=component_name
    )

def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger mit dem angegebenen Namen zurück.
    
    Args:
        name: Name des Loggers
        
    Returns:
        Logger-Instanz
    """
    log_manager = LogManager()
    return log_manager.get_logger(name)

def set_log_level(logger_name: Optional[str] = None, level: Optional[int] = None) -> bool:
    """
    Ändert das Log-Level eines Loggers zur Laufzeit.
    
    Args:
        logger_name: Name des Loggers
        level: Neues Log-Level
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    log_manager = LogManager()
    return log_manager.set_log_level(logger_name, level)

# Beispielverwendung
if __name__ == "__main__":
    # Beispiel für die Verwendung
    logger = setup_logging(component_name="example")
    logger.debug("Dies ist eine Debug-Nachricht")
    logger.info("Dies ist eine Info-Nachricht")
    logger.warning("Dies ist eine Warnung")
    logger.error("Dies ist ein Fehler")
    
    # Log-Level ändern
    set_log_level("example", logging.DEBUG)
    logger.debug("Diese Debug-Nachricht sollte jetzt angezeigt werden")
    
    # Module-Logger
    module_logger = get_logger("trading")
    module_logger.info("Nachricht aus dem Trading-Modul")

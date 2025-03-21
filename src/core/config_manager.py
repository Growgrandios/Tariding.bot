# config_manager.py

import os
import logging
import yaml
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Annahme: Logging wird bereits vom Hauptmodul eingerichtet
logger = logging.getLogger("ConfigManager")

class ConfigManager:
    """
    Zentrale Verwaltung aller Konfigurationseinstellungen des Trading Bots.
    Lädt Konfigurationen aus YAML-Dateien und .env-Dateien, verwaltet API-Schlüssel
    und stellt Konfigurationen für alle Module bereit.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den ConfigManager.
        
        Args:
            config_path: Optionaler Pfad zur Konfigurationsdatei (Standard: 'data/config/config.yaml')
        """
        logger.info("Initialisiere ConfigManager...")
        
        # Pfade
        self.config_path = config_path or 'data/config/config.yaml'
        self.config_dir = os.path.dirname(self.config_path)
        self.backup_dir = os.path.join(self.config_dir, 'backups')
        
        # Stelle sicher, dass die Verzeichnisse existieren
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Umgebungsvariablen laden
        load_dotenv()
        
        # Konfiguration laden
        self.config = self._load_config()
        
        # API-Schlüssel als separates Dictionary speichern
        self.api_keys = self._load_api_keys()
        
        logger.info("ConfigManager erfolgreich initialisiert")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus der YAML-Datei.
        
        Returns:
            Dictionary mit Konfigurationseinstellungen
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Konfiguration aus {self.config_path} geladen")
                return config or {}
            else:
                logger.warning(f"Konfigurationsdatei {self.config_path} nicht gefunden, verwende Standardwerte")
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Erstellt eine Standardkonfiguration und speichert sie.
        
        Returns:
            Dictionary mit Standardkonfiguration
        """
        default_config = {
            'general': {
                'bot_name': 'GemmaTrader',
                'log_level': 'INFO',
                'timezone': 'Europe/Berlin',
                'data_path': 'data',
                'version': '1.0.0'
            },
            'data_pipeline': {
                'update_intervals': {
                    'crypto': 60,
                    'stocks': 300,
                    'forex': 300,
                    'macro': 86400,
                    'news': 3600
                },
                'crypto_assets': [
                    'BTC/USDT:USDT',
                    'ETH/USDT:USDT',
                    'BNB/USDT:USDT',
                    'SOL/USDT:USDT',
                    'XRP/USDT:USDT'
                ],
                'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
            },
            'trading': {
                'mode': 'paper',  # 'paper', 'live', oder 'disabled'
                'default_leverage': 3,
                'max_leverage': 10,
                'risk_per_trade': 0.01,  # 1% des Kapitals pro Trade
                'max_open_trades': 3,
                'default_stop_loss_pct': 0.05,  # 5% Stop-Loss
                'default_take_profit_pct': 0.15,  # 15% Take-Profit
                'sandbox_mode': True
            },
            'black_swan_detector': {
                'volatility_threshold': 3.5,
                'volume_threshold': 5.0,
                'correlation_threshold': 0.85,
                'news_sentiment_threshold': -0.6,
                'check_interval': 300  # Sekunden
            },
            'telegram': {
                'enabled': True,
                'notification_level': 'INFO',
                'status_update_interval': 3600,  # Sekunden
                'commands_enabled': True
            },
            'learning_module': {
                'backtest_days': 90,
                'paper_trading_days': 14,
                'target_win_rate': 0.6,
                'training': {
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'patience': 10
                }
            },
            'tax_module': {
                'default_method': 'FIFO',  # FIFO, LIFO, HIFO
                'country': 'DE',
                'exempt_limit': 600  # Freigrenze in Euro
            }
        }
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(default_config, file, default_flow_style=False, sort_keys=False)
            logger.info(f"Standardkonfiguration in {self.config_path} gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Standardkonfiguration: {str(e)}")
        
        return default_config
    
    def _load_api_keys(self) -> Dict[str, Any]:
        """
        Lädt API-Schlüssel aus Umgebungsvariablen.
        
        Returns:
            Dictionary mit API-Schlüsseln
        """
        api_keys = {
            'bitget': {
                'api_key': os.getenv('BITGET_API_KEY', ''),
                'api_secret': os.getenv('BITGET_API_SECRET', ''),
                'api_passphrase': os.getenv('BITGET_API_PASSPHRASE', '')
            },
            'telegram': {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'allowed_users': os.getenv('TELEGRAM_ALLOWED_USERS', '')
            },
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'news_api': os.getenv('NEWS_API_KEY', '')
        }
        
        # Überprüfe, ob API-Schlüssel vorhanden sind
        missing_keys = []
        if not api_keys['bitget']['api_key']:
            missing_keys.append('BITGET_API_KEY')
        if not api_keys['bitget']['api_secret']:
            missing_keys.append('BITGET_API_SECRET')
        if not api_keys['bitget']['api_passphrase']:
            missing_keys.append('BITGET_API_PASSPHRASE')
        if not api_keys['telegram']['bot_token']:
            missing_keys.append('TELEGRAM_BOT_TOKEN')
        
        if missing_keys:
            logger.warning(f"Fehlende API-Schlüssel: {', '.join(missing_keys)}")
        else:
            logger.info("Alle erforderlichen API-Schlüssel geladen")
        
        return api_keys
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Gibt die Konfiguration zurück, optional für einen bestimmten Abschnitt.
        
        Args:
            section: Optionaler Konfigurationsabschnitt
        
        Returns:
            Komplette Konfiguration oder Konfiguration für einen bestimmten Abschnitt
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def get_api_keys(self) -> Dict[str, Any]:
        """
        Gibt alle API-Schlüssel zurück.
        
        Returns:
            Dictionary mit allen API-Schlüsseln
        """
        return self.api_keys
    
    def get_api_key(self, service: str) -> Dict[str, Any]:
        """
        Gibt API-Schlüssel für einen bestimmten Dienst zurück.
        
        Args:
            service: Name des Dienstes (z.B. 'bitget', 'telegram')
        
        Returns:
            API-Schlüssel für den angegebenen Dienst
        """
        return self.api_keys.get(service, {})
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Aktualisiert einen Konfigurationswert und speichert die Änderung.
        
        Args:
            section: Konfigurationsabschnitt
            key: Konfigurationsschlüssel
            value: Neuer Wert
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Erstelle Backup
            self._create_backup()
            
            # Konfiguration aktualisieren
            if section in self.config:
                self.config[section][key] = value
            else:
                self.config[section] = {key: value}
            
            # Konfiguration speichern
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Konfiguration aktualisiert: {section}.{key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Konfiguration: {str(e)}")
            return False
    
    def update_section(self, section: str, config: Dict[str, Any]) -> bool:
        """
        Aktualisiert einen gesamten Konfigurationsabschnitt und speichert die Änderung.
        
        Args:
            section: Konfigurationsabschnitt
            config: Neue Konfiguration für den Abschnitt
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Erstelle Backup
            self._create_backup()
            
            # Konfiguration aktualisieren
            self.config[section] = config
            
            # Konfiguration speichern
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Konfigurationsabschnitt {section} aktualisiert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des Konfigurationsabschnitts: {str(e)}")
            return False
    
    def _create_backup(self) -> str:
        """
        Erstellt ein Backup der aktuellen Konfigurationsdatei.
        
        Returns:
            Pfad zur Backup-Datei oder leerer String bei Fehler
        """
        try:
            if not os.path.exists(self.config_path):
                return ""
            
            # Erstelle Backup-Dateiname mit Zeitstempel
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"config_backup_{timestamp}.yaml"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Kopiere Konfigurationsdatei
            shutil.copy2(self.config_path, backup_path)
            
            # Alte Backups aufräumen (behalte max. 10 Backups)
            self._cleanup_backups(10)
            
            logger.info(f"Konfigurationsbackup erstellt: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Konfigurationsbackups: {str(e)}")
            return ""
    
    def _cleanup_backups(self, max_backups: int = 10) -> None:
        """
        Räumt alte Backup-Dateien auf, behält nur die neuesten.
        
        Args:
            max_backups: Maximale Anzahl an Backups, die behalten werden sollen
        """
        try:
            backup_files = sorted([
                os.path.join(self.backup_dir, f)
                for f in os.listdir(self.backup_dir)
                if f.startswith("config_backup_") and f.endswith(".yaml")
            ])
            
            # Lösche ältere Backups, wenn mehr als max_backups vorhanden sind
            if len(backup_files) > max_backups:
                files_to_delete = backup_files[:-max_backups]
                for file_path in files_to_delete:
                    os.remove(file_path)
                    logger.debug(f"Altes Konfigurationsbackup gelöscht: {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Aufräumen alter Backups: {str(e)}")
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Stellt eine Konfiguration aus einem Backup wieder her.
        
        Args:
            backup_path: Pfad zur Backup-Datei
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup-Datei nicht gefunden: {backup_path}")
                return False
            
            # Aktuelle Konfiguration sichern (für den Fall, dass etwas schief geht)
            self._create_backup()
            
            # Backup wiederherstellen
            shutil.copy2(backup_path, self.config_path)
            
            # Konfiguration neu laden
            self.config = self._load_config()
            
            logger.info(f"Konfiguration aus Backup wiederhergestellt: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Wiederherstellen der Konfiguration: {str(e)}")
            return False
    
    def get_backups(self) -> Dict[str, Any]:
        """
        Gibt eine Liste aller verfügbaren Konfigurationsbackups zurück.
        
        Returns:
            Dictionary mit Backup-Informationen
        """
        try:
            backups = []
            if os.path.exists(self.backup_dir):
                for filename in os.listdir(self.backup_dir):
                    if filename.startswith("config_backup_") and filename.endswith(".yaml"):
                        file_path = os.path.join(self.backup_dir, filename)
                        file_stats = os.stat(file_path)
                        
                        # Zeitstempel aus Dateinamen extrahieren
                        timestamp_str = filename.replace("config_backup_", "").replace(".yaml", "")
                        try:
                            timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except:
                            timestamp = datetime.datetime.fromtimestamp(file_stats.st_mtime)
                        
                        backups.append({
                            'filename': filename,
                            'path': file_path,
                            'timestamp': timestamp.isoformat(),
                            'size': file_stats.st_size
                        })
            
            # Nach Zeitstempel sortieren (neueste zuerst)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'count': len(backups),
                'backups': backups
            }
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Backup-Liste: {str(e)}")
            return {
                'count': 0,
                'backups': [],
                'error': str(e)
            }
    
    def export_config(self, export_path: str) -> bool:
        """
        Exportiert die aktuelle Konfiguration in eine Datei (ohne sensible Daten).
        
        Args:
            export_path: Pfad zur Export-Datei
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Kopie der Konfiguration erstellen
            config_copy = self._remove_sensitive_data(self.config)
            
            # Konfiguration exportieren
            with open(export_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_copy, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Konfiguration exportiert nach: {export_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Exportieren der Konfiguration: {str(e)}")
            return False
    
    def _remove_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entfernt sensible Daten aus einer Konfiguration.
        
        Args:
            config: Konfiguration
        
        Returns:
            Konfiguration ohne sensible Daten
        """
        import copy
        config_copy = copy.deepcopy(config)
        
        # Sensible Felder durch Platzhalter ersetzen
        sensitive_fields = [
            'api_key', 'api_secret', 'password', 'passphrase', 'secret',
            'token', 'access_token', 'refresh_token'
        ]
        
        def remove_sensitive(data):
            if isinstance(data, dict):
                for key, value in list(data.items()):
                    if isinstance(value, (dict, list)):
                        remove_sensitive(value)
                    elif any(sensitive in key.lower() for sensitive in sensitive_fields):
                        data[key] = "[REDACTED]"
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (dict, list)):
                        remove_sensitive(item)
        
        remove_sensitive(config_copy)
        return config_copy
    
    def import_config(self, import_path: str) -> bool:
        """
        Importiert eine Konfiguration aus einer Datei.
        
        Args:
            import_path: Pfad zur Import-Datei
        
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if not os.path.exists(import_path):
                logger.error(f"Import-Datei nicht gefunden: {import_path}")
                return False
            
            # Backup erstellen
            self._create_backup()
            
            # Konfiguration importieren
            with open(import_path, 'r', encoding='utf-8') as file:
                imported_config = yaml.safe_load(file)
            
            # Sensible Daten aus der aktuellen Konfiguration beibehalten
            merged_config = self._merge_configs(imported_config, self.config)
            
            # Aktualisierte Konfiguration speichern
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(merged_config, file, default_flow_style=False, sort_keys=False)
            
            # Konfiguration neu laden
            self.config = self._load_config()
            
            logger.info(f"Konfiguration importiert von: {import_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Importieren der Konfiguration: {str(e)}")
            return False
    
    def _merge_configs(self, imported: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt zwei Konfigurationen zusammen, wobei sensible Daten aus der aktuellen Konfiguration beibehalten werden.
        
        Args:
            imported: Importierte Konfiguration
            current: Aktuelle Konfiguration
        
        Returns:
            Zusammengeführte Konfiguration
        """
        import copy
        result = copy.deepcopy(imported)
        
        # Sensible Felder, die beibehalten werden sollen
        sensitive_fields = [
            'api_key', 'api_secret', 'password', 'passphrase', 'secret',
            'token', 'access_token', 'refresh_token'
        ]
        
        def merge_sensitive(imported_data, current_data, path=""):
            if isinstance(imported_data, dict) and isinstance(current_data, dict):
                for key, value in imported_data.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)) and key in current_data:
                        merge_sensitive(value, current_data[key], new_path)
                    elif any(sensitive in key.lower() for sensitive in sensitive_fields) and key in current_data:
                        # Behalte den Wert aus der aktuellen Konfiguration bei
                        imported_data[key] = current_data[key]
            elif isinstance(imported_data, list) and isinstance(current_data, list):
                # Bei Listen können wir nicht so einfach zusammenführen, daher nur sensible Objekte in der Liste prüfen
                for i, item in enumerate(imported_data):
                    if isinstance(item, (dict, list)) and i < len(current_data):
                        merge_sensitive(item, current_data[i], f"{path}[{i}]")
        
        merge_sensitive(result, current)
        return result
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Überprüft die Konfiguration auf Fehler oder fehlende Werte.
        
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        issues = []
        warnings = []
        
        # Überprüfe, ob alle erforderlichen Abschnitte vorhanden sind
        required_sections = ['general', 'data_pipeline', 'trading', 'black_swan_detector', 'telegram']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Fehlender Konfigurationsabschnitt: {section}")
        
        # Überprüfe API-Schlüssel
        if not self.api_keys['bitget']['api_key'] or not self.api_keys['bitget']['api_secret']:
            warnings.append("Bitget API-Schlüssel fehlen. Live-Trading ist nicht möglich.")
        if not self.api_keys['telegram']['bot_token']:
            warnings.append("Telegram Bot-Token fehlt. Telegram-Benachrichtigungen sind nicht möglich.")
        
        # Überprüfe Trading-Modus
        if 'trading' in self.config:
            trading_mode = self.config['trading'].get('mode', 'paper')
            if trading_mode == 'live' and (not self.api_keys['bitget']['api_key'] or not self.api_keys['bitget']['api_secret']):
                issues.append("Live-Trading-Modus aktiviert, aber Bitget API-Schlüssel fehlen.")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

# tax_module.py

import os
import logging
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple

class TaxMethod(Enum):
    """Methoden zur Berechnung von steuerrelevanten Gewinnen/Verlusten"""
    FIFO = "first_in_first_out"  # First In First Out
    LIFO = "last_in_first_out"   # Last In First Out
    HIFO = "highest_in_first_out"  # Highest In First Out

class TaxModule:
    """
    Modul für die Steuerberechnung und -dokumentation.
    
    Überwacht alle Trades, berechnet Gewinne und Verluste nach verschiedenen
    Methoden (FIFO, LIFO, HIFO) und erstellt Berichte für die Steuererklärung.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert das TaxModule.
        
        Args:
            config: Konfigurationseinstellungen
        """
        self.logger = logging.getLogger("TaxModule")
        self.logger.info("Initialisiere TaxModule...")
        
        # Konfiguration speichern
        self.config = config or {}
        
        # Basispfad für Steuerdaten
        self.base_path = Path(self.config.get('base_path', 'data/tax'))
        self.reports_path = self.base_path / 'reports'
        
        # Steuermethode festlegen
        method_str = self.config.get('default_method', 'FIFO')
        
        # Mapping von Konfigurationswerten zu Enum-Werten
        method_mapping = {
            'FIFO': TaxMethod.FIFO,
            'LIFO': TaxMethod.LIFO,
            'HIFO': TaxMethod.HIFO,
            'first_in_first_out': TaxMethod.FIFO,
            'last_in_first_out': TaxMethod.LIFO,
            'highest_in_first_out': TaxMethod.HIFO
        }
        
        self.method = method_mapping.get(method_str, TaxMethod.FIFO)
        
        # Steuerliche Einstellungen
        self.country = self.config.get('country', 'DE')
        self.tax_year = datetime.now().year
        self.exempt_limit = self.config.get('exempt_limit', 600)  # Freigrenze in Euro
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Historische Trades
        self.trades = []
        self.positions = {}
        
        # Aktuelle Steuerperiode
        self.current_period = {
            'year': datetime.now().year,
            'total_profit': 0,
            'total_loss': 0,
            'closed_positions': [],
            'open_positions': []
        }
        
        # Lade vorhandene Daten, falls vorhanden
        self._load_existing_data()
        
        self.logger.info("TaxModule erfolgreich initialisiert")
    
    def _load_existing_data(self):
        """
        Lädt vorhandene Handelsdaten und Steuerdaten, falls verfügbar.
        """
        try:
            # Pfad zur gespeicherten Handelsdaten
            trades_file = self.base_path / f"trades_{self.current_period['year']}.json"
            positions_file = self.base_path / f"positions_{self.current_period['year']}.json"
            
            # Lade Trades, falls vorhanden
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    self.trades = json.load(f)
                self.logger.info(f"Historische Trades geladen: {len(self.trades)} Einträge")
            
            # Lade Positionen, falls vorhanden
            if positions_file.exists():
                with open(positions_file, 'r') as f:
                    self.positions = json.load(f)
                self.logger.info(f"Offene Positionen geladen: {len(self.positions)} Einträge")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der bestehenden Daten: {str(e)}")
            # Setze auf Standardwerte bei Fehler
            self.trades = []
            self.positions = {}
    
    def process_trade(self, trade_data: Dict[str, Any]):
        """
        Verarbeitet einen neuen Trade für die Steuerberechnung.
        
        Args:
            trade_data: Dictionary mit Tradedaten
        """
        try:
            # Implementierung für die Verarbeitung von Trades
            self.logger.info(f"Trade verarbeitet: {trade_data.get('symbol', 'unbekannt')}")
            return True
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Trades: {str(e)}")
            return False
    
    def get_tax_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der aktuellen Steuersituation zurück.
        
        Returns:
            Dictionary mit Steuer-Zusammenfassung
        """
        try:
            # Basisimplementierung für Steuerzusammenfassung
            summary = {
                'year': self.current_period['year'],
                'tax_method': self.method.value,
                'country': self.country,
                'exempt_limit': self.exempt_limit,
                'timestamp': datetime.now().isoformat()
            }
            return summary
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Steuerzusammenfassung: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

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
    
    # [Rest der Klasse bleibt unverändert...]

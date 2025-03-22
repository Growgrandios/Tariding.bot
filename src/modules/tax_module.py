# tax_module.py

import os
import logging
import json
import csv
import pandas as pd
import numpy as np
import hashlib
import uuid
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from fpdf import FPDF
import xml.etree.ElementTree as ET

class TaxMethod(Enum):
    """Methoden zur Berechnung von steuerrelevanten Gewinnen/Verlusten"""
    FIFO = "first_in_first_out"  # First In First Out
    LIFO = "last_in_first_out"   # Last In First Out
    HIFO = "highest_in_first_out"  # Highest In First Out

class TaxStatus(Enum):
    """Steuerlicher Status der Gesamtposition"""
    BELOW_EXEMPT = "below_exempt_limit"  # Unter Freigrenze
    TAXABLE = "taxable"  # Steuerpflichtig
    HIGH_TAX_BURDEN = "high_tax_burden"  # Hohe Steuerlast

class SteuerId:
    """Validator und Konvertierer für Steuer-IDs und Steuernummern"""
    
    @staticmethod
    def validate_steuer_id(steuer_id: str) -> bool:
        """
        Prüft, ob eine Steuer-ID gültig ist (11-stellig mit korrekter Prüfziffer).
        
        Args:
            steuer_id: Die zu prüfende Steuer-ID
            
        Returns:
            True wenn gültig, sonst False
        """
        if not steuer_id or not isinstance(steuer_id, str):
            return False
            
        # Entferne Leerzeichen und Bindestriche
        steuer_id = steuer_id.replace(" ", "").replace("-", "")
        
        # Prüfe Länge und numerischen Inhalt
        if len(steuer_id) != 11 or not steuer_id.isdigit():
            return False
            
        # Prüfe, dass nicht alle Ziffern gleich sind
        if len(set(steuer_id)) == 1:
            return False
            
        # Berechne Prüfziffer
        pruefziffer = SteuerId.calc_pruefziffer(steuer_id[:10])
        
        # Vergleiche mit letzter Ziffer
        return pruefziffer == int(steuer_id[10])
    
    @staticmethod
    def calc_pruefziffer(steuer_id_base: str) -> int:
        """
        Berechnet die Prüfziffer einer Steuer-ID nach dem Modulo-11-Verfahren.
        
        Args:
            steuer_id_base: Die ersten 10 Ziffern der Steuer-ID
            
        Returns:
            Berechnete Prüfziffer
        """
        produkt = 10
        for ziffer in steuer_id_base:
            summe = (int(ziffer) + produkt) % 10
            if summe == 0:
                summe = 10
            produkt = (2 * summe) % 11
        
        pruefziffer = (11 - produkt) % 10
        return pruefziffer
    
    @staticmethod
    def convert_to_elster_format(steuer_nr: str, bundesland: str) -> str:
        """
        Konvertiert eine landesspezifische Steuernummer in das ELSTER-Format.
        
        Args:
            steuer_nr: Landesspezifische Steuernummer
            bundesland: Zweistelliger Bundeslandcode (z.B. 'NW' für NRW)
            
        Returns:
            Steuernummer im 13-stelligen ELSTER-Format
        """
        # Entferne Leerzeichen und Schrägstriche
        steuer_nr = steuer_nr.replace(" ", "").replace("/", "")
        
        # Bundesland-spezifische Konvertierung
        bundesland_map = {
            'NW': '5',  # Nordrhein-Westfalen
            'BY': '9',  # Bayern
            'BW': '8',  # Baden-Württemberg
            'NI': '2',  # Niedersachsen
            'HE': '6',  # Hessen
            'RP': '7',  # Rheinland-Pfalz
            'SN': '3',  # Sachsen
            'ST': '3',  # Sachsen-Anhalt
            'TH': '4',  # Thüringen
            'BB': '3',  # Brandenburg
            'MV': '4',  # Mecklenburg-Vorpommern
            'SH': '2',  # Schleswig-Holstein
            'SL': '1',  # Saarland
            'BE': '1',  # Berlin
            'HB': '2',  # Bremen
            'HH': '2',  # Hamburg
        }
        
        land_code = bundesland_map.get(bundesland, '0')
        
        # Implementierung für NRW als Beispiel
        if bundesland == 'NW':
            # Format: FFFF/BBBB/UUUP (F=Finanzamt, B=Bezirk, U=Unterscheidungsnummer, P=Prüfziffer)
            if len(steuer_nr) == 13:  # Bereits im Format FFFFBBBBVVVP
                return f"{land_code}{steuer_nr[0:4]}0{steuer_nr[4:12]}"
            else:
                # Annahme: FFFF/BBBB/UUUP
                parts = steuer_nr.split("/") if "/" in steuer_nr else [steuer_nr[:4], steuer_nr[4:8], steuer_nr[8:]]
                if len(parts) == 3:
                    return f"{land_code}{parts[0]}0{parts[1]}{parts[2]}"
        
        # Bayern
        elif bundesland == 'BY':
            # Format: FFF/BBB/UUUUP
            if len(steuer_nr) == 12:  # Bereits im Format FFFBBBVVVVP
                return f"{land_code}{steuer_nr[0:3]}0{steuer_nr[3:11]}"
            else:
                # Annahme: FFF/BBB/UUUUP
                parts = steuer_nr.split("/") if "/" in steuer_nr else [steuer_nr[:3], steuer_nr[3:6], steuer_nr[6:]]
                if len(parts) == 3:
                    return f"{land_code}{parts[0]}0{parts[1]}{parts[2]}"
        
        # Fallback: Direkte Übergabe mit Ländercode
        return f"{land_code}{steuer_nr}"

class TaxCalculator:
    """Berechnungslogik für Steuern und Abgaben"""
    
    @staticmethod
    def calculate_tax_rate(annual_income: float) -> float:
        """
        Berechnet den individuellen Steuersatz basierend auf dem Jahreseinkommen nach 2025er Tarif.
        
        Args:
            annual_income: Zu versteuerndes Jahreseinkommen in Euro
            
        Returns:
            Steuersatz als Dezimalwert (0.0 - 0.45)
        """
        # Einkommensteuer 2025 (vereinfacht)
        if annual_income <= 12084:
            return 0.0
        elif annual_income <= 17430:
            # Progressive Erhöhung von 14% auf 24%
            y = (annual_income - 12084) / 5346
            return 0.14 + y * 0.10
        elif annual_income <= 68430:
            # Progressive Erhöhung von 24% auf 42%
            y = (annual_income - 17430) / 51000
            return 0.24 + y * 0.18
        elif annual_income <= 277825:
            return 0.42
        else:
            return 0.45
    
    @staticmethod
    def calculate_solidarity_surcharge(tax_amount: float) -> float:
        """
        Berechnet den Solidaritätszuschlag auf den Steuerbetrag.
        
        Args:
            tax_amount: Berechneter Steuerbetrag in Euro
            
        Returns:
            Solidaritätszuschlag in Euro
        """
        # Freigrenze für Soli: 16.956 Euro Einkommensteuer
        if tax_amount <= 16956:
            return 0.0
        
        # Gleitzone für Soli
        elif tax_amount <= 31528:
            # Gleitzonenformel: (Steuerbetrag - 16.956) * 0.2
            return min((tax_amount - 16956) * 0.2, tax_amount * 0.055)
        
        # Regulärer Soli: 5,5% des Steuerbetrags
        else:
            return tax_amount * 0.055
    
    @staticmethod
    def calculate_church_tax(tax_amount: float, church_tax_rate: float = 0.09) -> float:
        """
        Berechnet die Kirchensteuer auf den Steuerbetrag.
        
        Args:
            tax_amount: Berechneter Steuerbetrag in Euro
            church_tax_rate: Kirchensteuersatz (0.08 für Bayern/BW, 0.09 für andere Bundesländer)
            
        Returns:
            Kirchensteuer in Euro
        """
        return tax_amount * church_tax_rate
    
    @staticmethod
    def calculate_crypto_tax(profit: float, tax_rate: float, exempt_limit: float = 1000.0) -> Dict[str, float]:
        """
        Berechnet die Steuer auf Kryptogewinne unter Berücksichtigung des Freibetrags.
        
        Args:
            profit: Bruttogewinn aus Kryptotransaktionen in Euro
            tax_rate: Individueller Steuersatz als Dezimalwert
            exempt_limit: Steuerfreier Betrag in Euro (Standard: 1000€, ab 2024)
            
        Returns:
            Dictionary mit Steuerdetails
        """
        # Wenn Gewinn unter Freigrenze, keine Steuer
        if profit <= exempt_limit:
            return {
                'taxable_profit': 0.0,
                'exempt_amount': profit,
                'tax_amount': 0.0,
                'effective_tax_rate': 0.0
            }
        
        # Ansonsten voller Betrag steuerpflichtig (Freigrenze, nicht Freibetrag)
        taxable_profit = profit
        tax_amount = taxable_profit * tax_rate
        
        return {
            'taxable_profit': taxable_profit,
            'exempt_amount': 0.0,  # Freigrenze, nicht Freibetrag
            'tax_amount': tax_amount,
            'effective_tax_rate': tax_amount / profit if profit > 0 else 0.0
        }

class TaxDocument:
    """Erzeugt steuerrelevante Dokumente und Berichte"""
    
    @staticmethod
    def generate_tax_report_pdf(tax_data: Dict[str, Any], output_path: str) -> bool:
        """
        Erzeugt einen PDF-Steuerbericht mit allen relevanten Informationen.
        
        Args:
            tax_data: Dictionary mit Steuerdaten
            output_path: Pfad zum Speichern der PDF-Datei
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Titel und Kopf
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Krypto-Steuerreport", 0, 1, "C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Steuerjahr: {tax_data.get('year', datetime.now().year)}", 0, 1)
            pdf.cell(0, 10, f"Erstellungsdatum: {datetime.now().strftime('%d.%m.%Y %H:%M')}", 0, 1)
            pdf.ln(5)
            
            # Zusammenfassung
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Zusammenfassung", 0, 1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Gesamtgewinn: {tax_data.get('total_profit', 0):.2f} EUR", 0, 1)
            pdf.cell(0, 10, f"Steuerpflichtiger Gewinn: {tax_data.get('taxable_profit', 0):.2f} EUR", 0, 1)
            pdf.cell(0, 10, f"Steuersatz: {tax_data.get('tax_rate', 0)*100:.1f}%", 0, 1)
            pdf.cell(0, 10, f"Zu zahlender Steuerbetrag: {tax_data.get('tax_amount', 0):.2f} EUR", 0, 1)
            
            # Rechtliche Hinweise
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Rechtliche Hinweise", 0, 1)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 10, "Dieser Bericht wurde automatisch erstellt und dient nur als Hilfestellung. "
                          "Für die endgültige Steuererklärung konsultieren Sie bitte einen Steuerberater. "
                          "Die Berechnung erfolgt nach § 23 EStG (private Veräußerungsgeschäfte).")
            
            # Speichern
            pdf.output(output_path)
            return True
            
        except Exception as e:
            logging.error(f"Fehler bei der PDF-Generierung: {str(e)}")
            return False
    
    @staticmethod
    def generate_sepa_xml(tax_payment: Dict[str, Any], output_path: str) -> bool:
        """
        Erzeugt eine SEPA-XML-Datei für die Steuerzahlung.
        
        Args:
            tax_payment: Dictionary mit Zahlungsdetails
            output_path: Pfad zum Speichern der XML-Datei
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Erstelle XML-Root
            root = ET.Element("Document")
            root.set("xmlns", "urn:iso:std:iso:20022:tech:xsd:pain.001.001.03")
            
            # CstmrCdtTrfInitn
            csti = ET.SubElement(root, "CstmrCdtTrfInitn")
            
            # GrpHdr
            grp_hdr = ET.SubElement(csti, "GrpHdr")
            ET.SubElement(grp_hdr, "MsgId").text = str(uuid.uuid4())
            ET.SubElement(grp_hdr, "CreDtTm").text = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            ET.SubElement(grp_hdr, "NbOfTxs").text = "1"
            ET.SubElement(grp_hdr, "CtrlSum").text = str(tax_payment.get('amount', 0))
            
            # PmtInf
            pmt_inf = ET.SubElement(csti, "PmtInf")
            ET.SubElement(pmt_inf, "PmtInfId").text = f"PMT-{datetime.now().strftime('%Y%m%d')}-1"
            ET.SubElement(pmt_inf, "PmtMtd").text = "TRF"
            ET.SubElement(pmt_inf, "ReqdExctnDt").text = tax_payment.get('execution_date', datetime.now().strftime("%Y-%m-%d"))
            
            # Debtor
            dbtr = ET.SubElement(pmt_inf, "Dbtr")
            ET.SubElement(dbtr, "Nm").text = tax_payment.get('debtor_name', 'Steuerzahler')
            
            # Schreibe in Datei
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding="UTF-8", xml_declaration=True)
            return True
            
        except Exception as e:
            logging.error(f"Fehler bei der SEPA-XML-Generierung: {str(e)}")
            return False

class CryptoPosition:
    """Repräsentiert eine Kryptoposition für die Steuerberechnung"""
    
    def __init__(self, symbol: str, amount: float, price: float, timestamp: datetime, 
                 transaction_id: str, transaction_type: str = "buy"):
        self.symbol = symbol
        self.amount = amount
        self.price = price
        self.timestamp = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp)
        self.transaction_id = transaction_id
        self.transaction_type = transaction_type  # "buy" oder "sell"
        self.remaining_amount = amount if transaction_type == "buy" else 0.0
        self.realized_profit = 0.0
        self.tax_relevant = self._is_tax_relevant()
    
    def _is_tax_relevant(self) -> bool:
        """
        Prüft, ob eine Position steuerrelevant ist (weniger als 1 Jahr gehalten).
        
        Returns:
            True wenn steuerrelevant, sonst False
        """
        holding_period = datetime.now() - self.timestamp
        return holding_period.days < 365
    
    def get_acquisition_cost(self) -> float:
        """
        Berechnet die Anschaffungskosten der Position.
        
        Returns:
            Anschaffungskosten in Euro
        """
        return self.price * self.amount
    
    def get_days_until_tax_free(self) -> int:
        """
        Berechnet die verbleibenden Tage bis zur Steuerfreiheit.
        
        Returns:
            Anzahl der Tage bis zum Ablauf der Spekulationsfrist
        """
        days_held = (datetime.now() - self.timestamp).days
        return max(0, 365 - days_held)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Wandelt das Objekt in ein Dictionary um.
        
        Returns:
            Position als Dictionary
        """
        return {
            'symbol': self.symbol,
            'amount': self.amount,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'transaction_id': self.transaction_id,
            'transaction_type': self.transaction_type,
            'remaining_amount': self.remaining_amount,
            'realized_profit': self.realized_profit,
            'tax_relevant': self.tax_relevant,
            'days_until_tax_free': self.get_days_until_tax_free()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoPosition':
        """
        Erstellt ein Objekt aus einem Dictionary.
        
        Args:
            data: Dictionary mit Positionsdaten
            
        Returns:
            CryptoPosition-Objekt
        """
        position = cls(
            symbol=data['symbol'],
            amount=data['amount'],
            price=data['price'],
            timestamp=data['timestamp'],
            transaction_id=data['transaction_id'],
            transaction_type=data['transaction_type']
        )
        position.remaining_amount = data.get('remaining_amount', position.amount)
        position.realized_profit = data.get('realized_profit', 0.0)
        position.tax_relevant = data.get('tax_relevant', position._is_tax_relevant())
        return position

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
        self.encrypted_path = self.base_path / 'encrypted'
        
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
        # Ab 2024 wurde die Freigrenze von €600 auf €1000 erhöht (§23 EStG)
        self.exempt_limit = 1000.0 if self.tax_year >= 2024 else 600.0
        self.annual_income = self.config.get('annual_income', 0)  # Geschätztes Jahreseinkommen
        self.church_tax_enabled = self.config.get('church_tax_enabled', False)
        self.church_tax_rate = self.config.get('church_tax_rate', 0.09)  # 9% Standard für die meisten Bundesländer
        self.solidarity_surcharge_enabled = self.config.get('solidarity_surcharge_enabled', True)
        
        # Verlustvorträge und -rückträge
        self.loss_carryforward = self.config.get('loss_carryforward', 0.0)
        self.loss_carryback = self.config.get('loss_carryback', 0.0)
        
        # Tracking für Swaps (Krypto-zu-Krypto-Tausch)
        self.swap_transactions = []
        
        # Verschlüsselungsschlüssel für sensible Daten
        self._init_encryption()
        
        # Steuernummer/ID
        self.tax_id = self.config.get('tax_id', "")
        self.tax_number = self.config.get('tax_number', "")
        self.bundesland = self.config.get('bundesland', "")
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.encrypted_path.mkdir(parents=True, exist_ok=True)
        
        # Positionen und Trades
        self.buys = {}  # symbol -> Liste von Kauf-Positionen
        self.sells = []  # Liste von Verkaufs-Transaktionen
        self.closed_positions = []  # Abgeschlossene Positionen
        
        # Aktuelle Steuerperiode
        self.current_period = {
            'year': datetime.now().year,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'taxable_profit': 0.0,
            'tax_amount': 0.0,
            'tax_status': TaxStatus.BELOW_EXEMPT.value,
            'last_update': datetime.now().isoformat()
        }
        
        # Steuersatz berechnen
        self._update_tax_rate()
        
        # Zahlungsplan
        self.payment_schedule = self._generate_payment_schedule()
        
        # Lade vorhandene Daten, falls vorhanden
        self._load_existing_data()
        
        self.logger.info("TaxModule erfolgreich initialisiert")
    
    def _init_encryption(self):
        """Initialisiert die Verschlüsselung für sensible Daten."""
        try:
            # Generiere oder lade Schlüssel
            key_file = self.base_path / '.key'
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generiere neuen Schlüssel
                salt = os.urandom(16)
                password = str(uuid.uuid4()).encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
                
                # Speichere Schlüssel
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                
                # Setze Berechtigungen (nur für aktuellen Benutzer lesbar)
                os.chmod(key_file, 0o600)
            
            # Erstelle Fernet-Instanz
            self.cipher = Fernet(self.encryption_key)
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Initialisierung der Verschlüsselung: {str(e)}")
            # Fallback: Dummy-Verschlüsselung
            self.encryption_key = b"dummy_key_for_testing_only_not_secure"
            self.cipher = None
    
    def _encrypt_data(self, data: str) -> str:
        """
        Verschlüsselt sensible Daten.
        
        Args:
            data: Zu verschlüsselnde Daten als String
            
        Returns:
            Verschlüsselte Daten als Base64-String
        """
        if not data:
            return ""
            
        try:
            if self.cipher:
                return self.cipher.encrypt(data.encode()).decode()
            else:
                # Dummy-Verschlüsselung
                return base64.b64encode(data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Fehler bei der Verschlüsselung: {str(e)}")
            return ""
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """
        Entschlüsselt sensible Daten.
        
        Args:
            encrypted_data: Verschlüsselte Daten als Base64-String
            
        Returns:
            Entschlüsselte Daten als String
        """
        if not encrypted_data:
            return ""
            
        try:
            if self.cipher:
                return self.cipher.decrypt(encrypted_data.encode()).decode()
            else:
                # Dummy-Entschlüsselung
                return base64.b64decode(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Fehler bei der Entschlüsselung: {str(e)}")
            return ""
    
    def _load_existing_data(self):
        """
        Lädt vorhandene Handelsdaten und Steuerdaten, falls verfügbar.
        """
        try:
            # Pfade zu gespeicherten Daten
            buys_file = self.base_path / f"buys_{self.current_period['year']}.json"
            sells_file = self.base_path / f"sells_{self.current_period['year']}.json"
            closed_file = self.base_path / f"closed_{self.current_period['year']}.json"
            period_file = self.base_path / f"period_{self.current_period['year']}.json"
            swaps_file = self.base_path / f"swaps_{self.current_period['year']}.json"
            
            # Lade Käufe, falls vorhanden
            if buys_file.exists():
                with open(buys_file, 'r') as f:
                    buys_data = json.load(f)
                    self.buys = {}
                    for symbol, positions in buys_data.items():
                        self.buys[symbol] = [CryptoPosition.from_dict(pos) for pos in positions]
                self.logger.info(f"Kauf-Positionen geladen: {sum(len(positions) for positions in self.buys.values())} Einträge")
            
            # Lade Verkäufe, falls vorhanden
            if sells_file.exists():
                with open(sells_file, 'r') as f:
                    self.sells = json.load(f)
                self.logger.info(f"Verkaufs-Transaktionen geladen: {len(self.sells)} Einträge")
            
            # Lade geschlossene Positionen, falls vorhanden
            if closed_file.exists():
                with open(closed_file, 'r') as f:
                    self.closed_positions = json.load(f)
                self.logger.info(f"Geschlossene Positionen geladen: {len(self.closed_positions)} Einträge")
            
            # Lade Swap-Transaktionen, falls vorhanden
            if swaps_file.exists():
                with open(swaps_file, 'r') as f:
                    self.swap_transactions = json.load(f)
                self.logger.info(f"Swap-Transaktionen geladen: {len(self.swap_transactions)} Einträge")
            
            # Lade Periodeninfo, falls vorhanden
            if period_file.exists():
                with open(period_file, 'r') as f:
                    self.current_period = json.load(f)
                self.logger.info(f"Periodeninfo geladen für Jahr {self.current_period['year']}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der bestehenden Daten: {str(e)}")
            # Setze auf Standardwerte bei Fehler
            self.buys = {}
            self.sells = []
            self.closed_positions = []
            self.swap_transactions = []
    
    def _save_data(self):
        """Speichert alle Daten in JSON-Dateien."""
        try:
            year = self.current_period['year']
            
            # Speichere Käufe
            buys_data = {}
            for symbol, positions in self.buys.items():
                buys_data[symbol] = [pos.to_dict() for pos in positions]
                
            with open(self.base_path / f"buys_{year}.json", 'w') as f:
                json.dump(buys_data, f, indent=2)
            
            # Speichere Verkäufe
            with open(self.base_path / f"sells_{year}.json", 'w') as f:
                json.dump(self.sells, f, indent=2)
            
            # Speichere geschlossene Positionen
            with open(self.base_path / f"closed_{year}.json", 'w') as f:
                json.dump(self.closed_positions, f, indent=2)
            
            # Speichere Swap-Transaktionen
            with open(self.base_path / f"swaps_{year}.json", 'w') as f:
                json.dump(self.swap_transactions, f, indent=2)
            
            # Speichere Periodeninfo
            self.current_period['last_update'] = datetime.now().isoformat()
            with open(self.base_path / f"period_{year}.json", 'w') as f:
                json.dump(self.current_period, f, indent=2)
            
            self.logger.info(f"Daten erfolgreich gespeichert für Jahr {year}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Daten: {str(e)}")
            return False
    
    def _update_tax_rate(self):
        """Aktualisiert den Steuersatz basierend auf dem geschätzten Jahreseinkommen."""
        if self.annual_income > 0:
            self.tax_rate = TaxCalculator.calculate_tax_rate(self.annual_income)
        else:
            # Standardwert: 25% (ähnlich Abgeltungssteuer)
            self.tax_rate = 0.25
        
        self.logger.info(f"Steuersatz aktualisiert: {self.tax_rate*100:.1f}%")
        self.current_period['tax_rate'] = self.tax_rate
    
    def set_annual_income(self, annual_income: float):
        """
        Setzt das geschätzte Jahreseinkommen für die Steuerberechnung.
        
        Args:
            annual_income: Geschätztes Jahreseinkommen in Euro
        """
        if annual_income >= 0:
            self.annual_income = annual_income
            self.config['annual_income'] = annual_income
            self._update_tax_rate()
            self._recalculate_tax()
            self._save_data()
            self.logger.info(f"Jahreseinkommen aktualisiert: {annual_income:.2f} €")
    
    def _generate_payment_schedule(self) -> List[Dict[str, Any]]:
        """
        Generiert einen Zahlungsplan für Steuervorauszahlungen.
        
        Returns:
            Liste von Zahlungsterminen mit Beträgen
        """
        year = self.current_period['year']
        schedule = []
        
        # Quartalszahlungen
        quarters = [
            {"name": "Q1", "date": f"{year}-03-10", "percentage": 0.25},
            {"name": "Q2", "date": f"{year}-06-10", "percentage": 0.25},
            {"name": "Q3", "date": f"{year}-09-10", "percentage": 0.25},
            {"name": "Q4", "date": f"{year}-12-10", "percentage": 0.25},
        ]
        
        for quarter in quarters:
            schedule.append({
                "name": f"Steuervorauszahlung {quarter['name']} {year}",
                "due_date": quarter['date'],
                "percentage": quarter['percentage'],
                "amount": 0.0,  # Wird später aktualisiert
                "status": "pending"
            })
        
        return schedule
    
    def update_payment_schedule(self):
        """Aktualisiert den Zahlungsplan basierend auf der aktuellen Steuerschätzung."""
        estimated_tax = self.current_period.get('tax_amount', 0.0)
        total_percentage = 0.0
        
        for payment in self.payment_schedule:
            # Nur ausstehende Zahlungen aktualisieren
            if payment['status'] == "pending":
                payment_percentage = payment['percentage']
                total_percentage += payment_percentage
                payment['amount'] = estimated_tax * payment_percentage
        
        self.logger.info(f"Zahlungsplan aktualisiert mit geschätzter Steuerlast {estimated_tax:.2f} €")
    
    def set_tax_id(self, tax_id: str, validate: bool = True):
        """
        Setzt die Steuer-ID des Nutzers.
        
        Args:
            tax_id: Steuer-ID (11-stellig)
            validate: Ob die ID validiert werden soll
        """
        if validate and not SteuerId.validate_steuer_id(tax_id):
            self.logger.error(f"Ungültige Steuer-ID: {tax_id}")
            return False
        
        # Verschlüssele und speichere
        self.tax_id = self._encrypt_data(tax_id)
        self.logger.info("Steuer-ID erfolgreich aktualisiert")
        return True
    
    def set_tax_number(self, tax_number: str, bundesland: str):
        """
        Setzt die Steuernummer des Nutzers.
        
        Args:
            tax_number: Steuernummer
            bundesland: Bundesland (2-stelliger Code)
        """
        self.tax_number = self._encrypt_data(tax_number)
        self.bundesland = bundesland
        self.logger.info("Steuernummer erfolgreich aktualisiert")
        return True
    
    def get_elster_formatted_tax_number(self) -> str:
        """
        Gibt die Steuernummer im ELSTER-Format zurück.
        
        Returns:
            Steuernummer im ELSTER-Format oder leerer String bei Fehler
        """
        if not self.tax_number:
            return ""
        
        try:
            decrypted_number = self._decrypt_data(self.tax_number)
            return SteuerId.convert_to_elster_format(decrypted_number, self.bundesland)
        except Exception as e:
            self.logger.error(f"Fehler bei der Konvertierung der Steuernummer: {str(e)}")
            return ""
    
    def process_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Verarbeitet einen neuen Trade für die Steuerberechnung.
        
        Args:
            trade_data: Dictionary mit Tradedaten
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Grundlegende Validierung
            required_fields = ['symbol', 'amount', 'price', 'timestamp', 'transaction_id', 'transaction_type']
            if not all(field in trade_data for field in required_fields):
                missing = [field for field in required_fields if field not in trade_data]
                self.logger.error(f"Fehlende Felder in Tradedaten: {missing}")
                return False
            
            symbol = trade_data['symbol']
            amount = float(trade_data['amount'])
            price = float(trade_data['price'])
            timestamp = trade_data['timestamp']
            transaction_id = trade_data['transaction_id']
            transaction_type = trade_data['transaction_type'].lower()
            
            # Datetime-Objekt erstellen, falls nötig
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # 1. Kauf verarbeiten
            if transaction_type == "buy" and amount > 0:
                position = CryptoPosition(
                    symbol=symbol,
                    amount=amount,
                    price=price,
                    timestamp=timestamp,
                    transaction_id=transaction_id,
                    transaction_type="buy"
                )
                
                # Zur Liste der Käufe hinzufügen
                if symbol not in self.buys:
                    self.buys[symbol] = []
                self.buys[symbol].append(position)
                
                self.logger.info(f"Kaufposition hinzugefügt: {symbol}, {amount} @ {price} €")
                self._save_data()
                return True
            
            # 2. Verkauf verarbeiten
            elif transaction_type == "sell" and amount > 0:
                # Prüfen, ob genug Kauf-Positionen vorhanden sind
                if symbol not in self.buys or not self.buys[symbol]:
                    self.logger.error(f"Keine Kauf-Positionen für {symbol} vorhanden")
                    return False
                
                # Verkaufstransaktion speichern
                sell_transaction = {
                    'symbol': symbol,
                    'amount': amount,
                    'price': price,
                    'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                    'transaction_id': transaction_id
                }
                self.sells.append(sell_transaction)
                
                # Gewinne/Verluste berechnen basierend auf der Steuermethode
                result = self._match_position_for_sale(symbol, amount, price, timestamp)
                
                # Steuern neu berechnen
                self._recalculate_tax()
                
                self.logger.info(f"Verkauf verarbeitet: {symbol}, {amount} @ {price} €, " +
                              f"Gewinn/Verlust: {result.get('realized_profit', 0):.2f} €")
                
                # Daten speichern
                self._save_data()
                return True
            
            # 3. Swap (Krypto-zu-Krypto-Tausch) verarbeiten
            elif transaction_type == "exchange":
                return self._process_swap(trade_data)
            
            else:
                self.logger.error(f"Ungültiger Transaktionstyp: {transaction_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Trades: {str(e)}")
            return False
    
    def _process_swap(self, trade_data: Dict[str, Any]) -> bool:
        """
        Verarbeitet Krypto-zu-Krypto-Tausch gemäß §23 EStG.
        
        Args:
            trade_data: Dictionary mit Swap-Daten
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Validiere erforderliche Felder
            required = ['from_currency', 'to_currency', 'from_amount', 'to_amount',
                       'from_eur_value', 'to_eur_value', 'timestamp', 'transaction_id']
            if not all(k in trade_data for k in required):
                missing = [k for k in required if k not in trade_data]
                self.logger.error(f"Fehlende Felder für Swap: {missing}")
                return False

            # 1. Verkauf der Ursprungswährung (realisierter Gewinn/Verlust)
            sell_transaction = {
                'symbol': trade_data['from_currency'],
                'amount': trade_data['from_amount'],
                'price': trade_data['from_eur_value'] / trade_data['from_amount'],
                'timestamp': trade_data['timestamp'],
                'transaction_id': f"{trade_data['transaction_id']}_sell",
                'transaction_type': 'sell',
                'is_swap': True
            }
            sell_result = self.process_trade(sell_transaction)
            
            if not sell_result:
                self.logger.error(f"Fehler beim Verarbeiten des Verkaufs im Swap")
                return False

            # 2. Kauf der Zielwährung
            buy_transaction = {
                'symbol': trade_data['to_currency'],
                'amount': trade_data['to_amount'],
                'price': trade_data['to_eur_value'] / trade_data['to_amount'],
                'timestamp': trade_data['timestamp'],
                'transaction_id': f"{trade_data['transaction_id']}_buy",
                'transaction_type': 'buy',
                'is_swap': True
            }
            buy_result = self.process_trade(buy_transaction)
            
            if not buy_result:
                self.logger.error(f"Fehler beim Verarbeiten des Kaufs im Swap")
                return False

            # Protokolliere Swap
            swap_record = {
                'timestamp': trade_data['timestamp'] if isinstance(trade_data['timestamp'], str) else trade_data['timestamp'].isoformat(),
                'transaction_id': trade_data['transaction_id'],
                'from_currency': trade_data['from_currency'],
                'to_currency': trade_data['to_currency'],
                'from_amount': trade_data['from_amount'],
                'to_amount': trade_data['to_amount'],
                'from_eur_value': trade_data['from_eur_value'],
                'to_eur_value': trade_data['to_eur_value'],
                'swap_fee': trade_data.get('fee_eur', 0)
            }
            self.swap_transactions.append(swap_record)
            
            # Daten speichern
            self._save_data()
            self.logger.info(f"Swap verarbeitet: {trade_data['from_currency']} zu {trade_data['to_currency']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler bei Swap-Verarbeitung: {str(e)}")
            return False
    
    def _match_position_for_sale(self, symbol: str, sell_amount: float, sell_price: float, 
                               timestamp) -> Dict[str, Any]:
        """
        Ordnet einen Verkauf den entsprechenden Kaufpositionen zu und berechnet Gewinne/Verluste.
        
        Args:
            symbol: Gehandeltes Symbol
            sell_amount: Verkaufte Menge
            sell_price: Verkaufspreis
            timestamp: Zeitstempel des Verkaufs
            
        Returns:
            Dictionary mit Verkaufsergebnis
        """
        # Umwandeln in datetime-Objekt
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Sicherstellen, dass Kaufpositionen vorhanden sind
        if symbol not in self.buys or not self.buys[symbol]:
            return {'error': f'Keine Kaufpositionen für {symbol} gefunden'}
        
        # Steuermethode bestimmen
        if self.method == TaxMethod.FIFO:
            # Älteste zuerst (First In First Out)
            positions = sorted(self.buys[symbol], key=lambda pos: pos.timestamp)
        elif self.method == TaxMethod.LIFO:
            # Neueste zuerst (Last In First Out)
            positions = sorted(self.buys[symbol], key=lambda pos: pos.timestamp, reverse=True)
        elif self.method == TaxMethod.HIFO:
            # Teuerste zuerst (Highest In First Out)
            positions = sorted(self.buys[symbol], key=lambda pos: pos.price, reverse=True)
        else:
            # Fallback zu FIFO
            positions = sorted(self.buys[symbol], key=lambda pos: pos.timestamp)
        
        remaining_sell_amount = sell_amount
        realized_profit = 0.0
        taxable_profit = 0.0
        closed_positions = []
        remaining_positions = []
        
        # Verkauf auf Kaufpositionen aufteilen
        for pos in positions:
            if remaining_sell_amount <= 0:
                # Keine weiteren Verkäufe mehr übrig
                remaining_positions.append(pos)
                continue
            
            if pos.remaining_amount <= 0:
                # Position bereits vollständig verkauft
                continue
            
            # Bestimme die zu verkaufende Menge aus dieser Position
            sell_from_position = min(pos.remaining_amount, remaining_sell_amount)
            remaining_sell_amount -= sell_from_position
            
            # Berechne den Gewinn/Verlust für diesen Teil
            acquisition_cost = pos.price * sell_from_position
            sale_proceeds = sell_price * sell_from_position
            position_profit = sale_proceeds - acquisition_cost
            
            # Bestimme, ob der Gewinn/Verlust steuerpflichtig ist
            is_taxable = pos._is_tax_relevant()
            
            # Protokolliere die geschlossene Position
            closed_position = {
                'symbol': symbol,
                'buy_amount': sell_from_position,
                'buy_price': pos.price,
                'buy_date': pos.timestamp.isoformat(),
                'sell_price': sell_price,
                'sell_date': timestamp.isoformat(),
                'holding_period_days': (timestamp - pos.timestamp).days,
                'profit': position_profit,
                'taxable': is_taxable,
                'transaction_id': pos.transaction_id
            }
            closed_positions.append(closed_position)
            self.closed_positions.append(closed_position)
            
            # Aktualisiere die verbleibende Menge in der Kaufposition
            pos.remaining_amount -= sell_from_position
            pos.realized_profit += position_profit
            
            # Aktualisiere die Gesamtergebnisse
            realized_profit += position_profit
            if is_taxable:
                taxable_profit += position_profit
            
            # Wenn noch etwas übrig ist, behalte die Position
            if pos.remaining_amount > 0:
                remaining_positions.append(pos)
        
        # Aktualisiere die Kaufpositionen
        self.buys[symbol] = [pos for pos in positions if pos.remaining_amount > 0]
        
        # Aktualisiere die Periodenstatistik
        if realized_profit > 0:
            self.current_period['total_profit'] += realized_profit
        else:
            self.current_period['total_loss'] += abs(realized_profit)
        
        return {
            'realized_profit': realized_profit,
            'taxable_profit': taxable_profit,
            'closed_positions': closed_positions,
            'remaining_sell_amount': remaining_sell_amount
        }
    
    def _recalculate_tax(self):
        """Berechnet die Steuern basierend auf den aktuellen Positionen neu."""
        # 1. Gesamtgewinne und -verluste ermitteln
        total_profit = self.current_period['total_profit']
        total_loss = self.current_period['total_loss']
        
        # 2. Verluste mit Gewinnen verrechnen
        net_profit = max(0, total_profit - total_loss)
        
        # 3. Vorhandene Verlustvorträge berücksichtigen
        if self.loss_carryforward < 0 and net_profit > 0:
            # Nutze so viel vom Verlustvortrag wie möglich
            usable_loss = min(abs(self.loss_carryforward), net_profit)
            net_profit -= usable_loss
            self.loss_carryforward += usable_loss
            self.logger.info(f"Verlustvortrag in Höhe von {usable_loss:.2f} € angewendet")
        
        # 4. Steuerpflichtigen Gewinn berechnen (unter Berücksichtigung der Freigrenze)
        tax_calculation = TaxCalculator.calculate_crypto_tax(
            profit=net_profit, 
            tax_rate=self.tax_rate,
            exempt_limit=self.exempt_limit
        )
        
        # 5. Zusätzliche Abgaben berechnen
        tax_amount = tax_calculation['tax_amount']
        soli_amount = 0.0
        church_tax_amount = 0.0
        
        if self.solidarity_surcharge_enabled:
            soli_amount = TaxCalculator.calculate_solidarity_surcharge(tax_amount)
        
        if self.church_tax_enabled:
            church_tax_amount = TaxCalculator.calculate_church_tax(tax_amount, self.church_tax_rate)
        
        # 6. Aktuelle Verluste als Verlustvortrag für nächstes Jahr speichern
        if net_profit == 0 and total_loss > 0:
            # Wir haben mehr Verluste als Gewinne
            additional_loss = total_loss - total_profit
            self.loss_carryforward -= additional_loss
            self.logger.info(f"Neuer Verlustvortrag: {self.loss_carryforward:.2f} €")
        
        # 7. Aktualisiere die Periodenstatistik
        self.current_period['net_profit'] = net_profit
        self.current_period['taxable_profit'] = tax_calculation['taxable_profit']
        self.current_period['tax_amount'] = tax_amount
        self.current_period['solidarity_surcharge'] = soli_amount
        self.current_period['church_tax'] = church_tax_amount
        self.current_period['total_tax_burden'] = tax_amount + soli_amount + church_tax_amount
        self.current_period['loss_carryforward'] = self.loss_carryforward
        
        # 8. Aktualisiere den Steuerstatus
        if net_profit <= self.exempt_limit:
            self.current_period['tax_status'] = TaxStatus.BELOW_EXEMPT.value
        elif tax_amount < 5000:
            self.current_period['tax_status'] = TaxStatus.TAXABLE.value
        else:
            self.current_period['tax_status'] = TaxStatus.HIGH_TAX_BURDEN.value
        
        # 9. Aktualisiere den Zahlungsplan
        self.update_payment_schedule()
        
        self.logger.info(f"Steuerberechnung aktualisiert: Nettogewinn {net_profit:.2f} €, " +
                     f"Steuerpflichtiger Gewinn {tax_calculation['taxable_profit']:.2f} €, " +
                     f"Steuerbetrag {tax_amount:.2f} €")
    
    def apply_loss_carryback(self, previous_year_taxes: float) -> float:
        """
        Wendet Verlustrücktrag auf Vorjahressteuer an.
        
        Args:
            previous_year_taxes: Gezahlte Steuer aus Vorjahr
            
        Returns:
            Erstattungsbetrag
        """
        if self.loss_carryforward >= 0:
            return 0.0
        
        # Bestimme den nutzbaren Verlust (max. die gezahlte Steuer)
        usable_loss = min(abs(self.loss_carryforward), previous_year_taxes / self.tax_rate)
        refund_amount = usable_loss * self.tax_rate
        
        # Aktualisiere den Verlustvortrag
        self.loss_carryforward += usable_loss
        
        self.logger.info(f"Verlustrücktrag in Höhe von {usable_loss:.2f} € " +
                      f"angewendet, Erstattung: {refund_amount:.2f} €")
        
        return refund_amount
    
    def get_loss_utilization(self) -> Dict[str, Any]:
        """
        Gibt Verlustverrechnungsstatus zurück.
        
        Returns:
            Dictionary mit Verlustverrechnungsinformationen
        """
        return {
            'current_losses': self.current_period.get('total_loss', 0),
            'current_profits': self.current_period.get('total_profit', 0),
            'net_profit': self.current_period.get('net_profit', 0),
            'loss_carryforward': self.loss_carryforward,
            'loss_carryback': self.loss_carryback,
            'remaining_losses': (self.loss_carryforward + self.loss_carryback)
        }
    
    def get_tax_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der aktuellen Steuersituation zurück.
        
        Returns:
            Dictionary mit Steuer-Zusammenfassung
        """
        try:
            # Aktuelle Steuersituation
            summary = {
                'year': self.current_period['year'],
                'tax_method': self.method.value,
                'country': self.country,
                'exempt_limit': self.exempt_limit,
                'total_profit': self.current_period.get('total_profit', 0),
                'total_loss': self.current_period.get('total_loss', 0),
                'net_profit': self.current_period.get('net_profit', 0),
                'taxable_profit': self.current_period.get('taxable_profit', 0),
                'tax_rate': self.current_period.get('tax_rate', self.tax_rate),
                'tax_amount': self.current_period.get('tax_amount', 0),
                'solidarity_surcharge': self.current_period.get('solidarity_surcharge', 0),
                'church_tax': self.current_period.get('church_tax', 0),
                'total_tax_burden': self.current_period.get('total_tax_burden', 0),
                'tax_status': self.current_period.get('tax_status', TaxStatus.BELOW_EXEMPT.value),
                'loss_carryforward': self.loss_carryforward,
                'timestamp': datetime.now().isoformat(),
                'legal_basis': "§ 23 EStG (Private Veräußerungsgeschäfte)"
            }
            
            # Füge den Zahlungsplan hinzu
            summary['payment_schedule'] = self.payment_schedule
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Steuerzusammenfassung: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung aller offenen Positionen zurück.
        
        Returns:
            Dictionary mit Positionsübersicht
        """
        try:
            positions_summary = {
                'open_positions': [],
                'total_open_positions': 0,
                'total_purchase_value': 0.0,
                'total_current_value': 0.0,
                'unrealized_profit': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Offene Positionen sammeln
            for symbol, positions in self.buys.items():
                for pos in positions:
                    if pos.remaining_amount > 0:
                        # Aktuellen Wert schätzen (hier: Kaufpreis als Platzhalter)
                        current_price = pos.price  # In einer realen Anwendung würde man den aktuellen Kurs abrufen
                        current_value = pos.remaining_amount * current_price
                        purchase_value = pos.remaining_amount * pos.price
                        unrealized_profit = current_value - purchase_value
                        
                        position_info = {
                            'symbol': symbol,
                            'amount': pos.remaining_amount,
                            'purchase_price': pos.price,
                            'purchase_date': pos.timestamp.isoformat(),
                            'purchase_value': purchase_value,
                            'current_price': current_price,
                            'current_value': current_value,
                            'unrealized_profit': unrealized_profit,
                            'holding_period_days': (datetime.now() - pos.timestamp).days,
                            'days_until_tax_free': pos.get_days_until_tax_free(),
                            'tax_relevant': pos.tax_relevant
                        }
                        
                        positions_summary['open_positions'].append(position_info)
                        positions_summary['total_open_positions'] += 1
                        positions_summary['total_purchase_value'] += purchase_value
                        positions_summary['total_current_value'] += current_value
                        positions_summary['unrealized_profit'] += unrealized_profit
            
            return positions_summary
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Positionsübersicht: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_tax_report(self, output_format: str = "pdf") -> str:
        """
        Erzeugt einen detaillierten Steuerbericht im angegebenen Format.
        
        Args:
            output_format: Format des Berichts ("pdf", "csv" oder "json")
            
        Returns:
            Pfad zur erzeugten Berichtsdatei oder Fehlermeldung
        """
        try:
            # Aktuelle Steuerinformationen abrufen
            tax_summary = self.get_tax_summary()
            positions_summary = self.get_positions_summary()
            loss_utilization = self.get_loss_utilization()
            
            # Kombiniere die Daten
            report_data = {
                **tax_summary,
                'positions': positions_summary['open_positions'],
                'closed_positions': self.closed_positions,
                'swap_transactions': self.swap_transactions,
                'loss_carryforward': self.loss_carryforward,
                'loss_utilization': loss_utilization,
                'tax_free_allowance_used': min(self.exempt_limit, tax_summary.get('net_profit', 0))
            }
            
            # Erzeuge Dateinamen mit Zeitstempel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"steuerreport_{self.current_period['year']}_{timestamp}"
            
            if output_format == "pdf":
                output_path = self.reports_path / f"{filename}.pdf"
                success = TaxDocument.generate_tax_report_pdf(report_data, str(output_path))
                if success:
                    return str(output_path)
                else:
                    return "Fehler bei der PDF-Generierung"
                    
            elif output_format == "csv":
                output_path = self.reports_path / f"{filename}.csv"
                
                # Schreibe Transaktionsdaten in CSV
                with open(output_path, 'w', newline='') as csvfile:
                    fieldnames = ['symbol', 'buy_date', 'buy_price', 'buy_amount', 
                                'sell_date', 'sell_price', 'profit', 'taxable']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for position in self.closed_positions:
                        writer.writerow({
                            'symbol': position['symbol'],
                            'buy_date': position['buy_date'],
                            'buy_price': position['buy_price'],
                            'buy_amount': position['buy_amount'],
                            'sell_date': position['sell_date'],
                            'sell_price': position['sell_price'],
                            'profit': position['profit'],
                            'taxable': position['taxable']
                        })
                
                return str(output_path)
                
            elif output_format == "json":
                output_path = self.reports_path / f"{filename}.json"
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                return str(output_path)
                
            else:
                return f"Nicht unterstütztes Format: {output_format}"
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Berichtsgenerierung: {str(e)}")
            return f"Fehler: {str(e)}"
    
    def generate_payment_document(self, quarter: int) -> str:
        """
        Erzeugt ein Zahlungsdokument für die Steuervorauszahlung eines Quartals.
        
        Args:
            quarter: Quartal (1-4)
            
        Returns:
            Pfad zur erzeugten Zahlungsdatei oder Fehlermeldung
        """
        try:
            if quarter < 1 or quarter > 4:
                return f"Ungültiges Quartal: {quarter}"
                
            # Zahlungsdaten abrufen
            if len(self.payment_schedule) < quarter:
                return "Zahlungsplan enthält nicht genügend Einträge"
                
            payment_data = self.payment_schedule[quarter - 1]
            
            # Prüfe, ob eine Zahlung fällig ist
            if payment_data['amount'] <= 0:
                return "Keine Zahlung für dieses Quartal fällig"
                
            # Erzeuge SEPA-XML
            payment_info = {
                'amount': payment_data['amount'],
                'execution_date': payment_data['due_date'],
                'debtor_name': self.config.get('name', "Steuerzahler"),
                'reference': f"Steuervorauszahlung Q{quarter} {self.current_period['year']}"
            }
            
            # Ausgabepfad
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.reports_path / f"steuer_sepa_q{quarter}_{timestamp}.xml"
            
            success = TaxDocument.generate_sepa_xml(payment_info, str(output_path))
            if success:
                # Markiere Zahlung als vorbereitet
                payment_data['status'] = "prepared"
                payment_data['document_path'] = str(output_path)
                self._save_data()
                return str(output_path)
            else:
                return "Fehler bei der SEPA-XML-Generierung"
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Zahlungsdokumenterstellung: {str(e)}")
            return f"Fehler: {str(e)}"
    
    def visualize_tax_status(self) -> Dict[str, Any]:
        """
        Erzeugt eine visuelle Darstellung des aktuellen Steuerstatus.
        
        Returns:
            Dictionary mit Visualisierungsdaten
        """
        try:
            # Daten für die Visualisierung
            net_profit = self.current_period.get('net_profit', 0)
            exempt_limit = self.exempt_limit
            tax_amount = self.current_period.get('tax_amount', 0)
            
            # Bestimme den Steuerstatus
            if net_profit <= exempt_limit:
                status = "green"  # Unter Freigrenze
                percent_of_limit = net_profit / exempt_limit * 100
            elif tax_amount < 5000:
                status = "yellow"  # Moderate Steuerlast
                percent_of_limit = min(100, tax_amount / 5000 * 100)
            else:
                status = "red"  # Hohe Steuerlast
                percent_of_limit = 100
            
            # Ergebnisse
            visualization = {
                'status': status,
                'percent_of_limit': percent_of_limit,
                'net_profit': net_profit,
                'exempt_limit': exempt_limit,
                'tax_amount': tax_amount,
                'visualization_type': 'tax_status',
                'timestamp': datetime.now().isoformat()
            }
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Statusvisualisierung: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_tax_optimization(self) -> Dict[str, Any]:
        """
        Analysiert mögliche Steueroptimierungen basierend auf den offenen Positionen.
        
        Returns:
            Dictionary mit Optimierungsvorschlägen
        """
        try:
            optimization_results = {
                'positions_near_1_year': [],
                'high_profit_positions': [],
                'loss_positions': [],
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Alle offenen Positionen durchgehen
            for symbol, positions in self.buys.items():
                for pos in positions:
                    if pos.remaining_amount <= 0:
                        continue
                        
                    days_held = (datetime.now() - pos.timestamp).days
                    days_until_tax_free = max(0, 365 - days_held)
                    
                    # Position nah an der 1-Jahres-Grenze (7-30 Tage)
                    if 7 <= days_until_tax_free <= 30:
                        optimization_results['positions_near_1_year'].append({
                            'symbol': symbol,
                            'amount': pos.remaining_amount,
                            'days_until_tax_free': days_until_tax_free,
                            'purchase_date': pos.timestamp.isoformat(),
                            'purchase_price': pos.price
                        })
                        
                        # Empfehlung hinzufügen
                        optimization_results['recommendations'].append({
                            'type': 'hold_until_tax_free',
                            'symbol': symbol,
                            'amount': pos.remaining_amount,
                            'days_to_wait': days_until_tax_free,
                            'tax_free_date': (pos.timestamp + timedelta(days=365)).isoformat(),
                            'potential_tax_saving': pos.price * pos.remaining_amount * self.tax_rate * 0.5  # Geschätzte Ersparnis
                        })
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Steueroptimierungsanalyse: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_closing_date(self) -> str:
        """
        Gibt das Datum zurück, bis zu dem die Steuererklärung eingereicht werden muss.
        
        Returns:
            Datum im ISO-Format
        """
        # Steuererklärungen für ein Jahr müssen in Deutschland bis zum 31. Juli des Folgejahres eingereicht werden
        year = self.current_period['year']
        closing_date = datetime(year + 1, 7, 31).isoformat()
        
        return closing_date
    
    def reset_yearly_data(self):
        """Setzt die Daten für ein neues Steuerjahr zurück."""
        # Alte Daten sichern
        old_year = self.current_period['year']
        self._save_data()
        
        # Auf neues Jahr umstellen
        self.current_period = {
            'year': datetime.now().year,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'taxable_profit': 0.0,
            'tax_amount': 0.0,
            'tax_status': TaxStatus.BELOW_EXEMPT.value,
            'last_update': datetime.now().isoformat()
        }
        
        # Nur die abgeschlossenen Positionen zurücksetzen, offene behalten
        self.closed_positions = []
        self.sells = []
        
        # Neuen Zahlungsplan erstellen
        self.payment_schedule = self._generate_payment_schedule()
        
        # Speichern
        self._save_data()
        
        self.logger.info(f"Daten für neues Steuerjahr {self.current_period['year']} zurückgesetzt")

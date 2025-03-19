# tax_module.py

import os
import csv
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tax_module.log"),
        logging.StreamHandler()
    ]
)

class TaxMethod(Enum):
    """
    Unterstützte Methoden für die Berechnung des steuerpflichtigen Gewinns.
    """
    FIFO = "first_in_first_out"  # Erste gekaufte Assets werden zuerst verkauft
    LIFO = "last_in_first_out"   # Letzte gekaufte Assets werden zuerst verkauft
    HIFO = "highest_in_first_out"  # Teuerste Assets werden zuerst verkauft

class TaxModule:
    """
    Verfolgt und berechnet Steuerinformationen für Trading-Aktivitäten.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das Steuermodul.
        
        Args:
            config: Konfigurationseinstellungen für das Steuermodul
        """
        self.logger = logging.getLogger("TaxModule")
        self.logger.info("Initialisiere TaxModule...")
        
        # Konfiguration laden
        self.config = config or {}
        self.method = TaxMethod(self.config.get('default_method', 'first_in_first_out'))
        self.country = self.config.get('country', 'DE')
        
        # Steuerparameter basierend auf dem Land
        self.tax_parameters = self._get_tax_parameters()
        self.exempt_limit = self.config.get('exempt_limit', self.tax_parameters.get('exempt_limit', 0))
        
        # Pfade für Daten
        self.data_path = Path(self.config.get('data_path', 'data/tax'))
        self.trades_file = self.data_path / 'trades.csv'
        self.positions_file = self.data_path / 'positions.json'
        self.reports_path = self.data_path / 'reports'
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Datenstrukturen für Trades und Positionen
        self.trades = []
        self.positions = {}  # Symbol -> Liste von Positionen
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        # Lade existierende Daten, falls vorhanden
        self._load_trades()
        self._load_positions()
        
        self.logger.info(f"TaxModule initialisiert mit Methode {self.method.value} für Land {self.country}")
    
    def _get_tax_parameters(self) -> Dict[str, Any]:
        """
        Gibt länderspezifische Steuerparameter zurück.
        
        Returns:
            Dictionary mit Steuerparametern für das konfigurierte Land
        """
        # Steuerparameter für verschiedene Länder
        tax_params = {
            'DE': {
                'tax_rate': 0.25,  # 25% Abgeltungssteuer
                'solidarity_surcharge': 0.055,  # 5.5% Solidaritätszuschlag auf die Steuer
                'church_tax': 0.08,  # 8-9% Kirchensteuer (falls zutreffend)
                'exempt_limit': 600,  # 600€ Freigrenze
                'loss_offset': True,  # Verlustverrechnung möglich
                'holding_period': None  # Keine Haltefrist für Krypto (in Deutschland)
            },
            'US': {
                'short_term_rate': 0.22,  # Kurzfristiger Steuersatz (abhängig vom Einkommen)
                'long_term_rate': 0.15,  # Langfristiger Steuersatz (abhängig vom Einkommen)
                'exempt_limit': 0,  # Keine allgemeine Freigrenze
                'loss_offset_limit': 3000,  # Verlustverrechnung bis $3000 pro Jahr
                'holding_period': 365  # 1 Jahr Haltefrist für langfristige Kapitalerträge
            },
            'UK': {
                'tax_rate': 0.20,  # 20% Grundsteuersatz (abhängig vom Einkommen)
                'higher_rate': 0.40,  # 40% höherer Steuersatz
                'exempt_limit': 12300,  # £12,300 Capital Gains Allowance
                'loss_offset': True,
                'holding_period': None
            }
        }
        
        # Standardmäßig deutsche Steuerparameter zurückgeben, falls Land nicht bekannt
        return tax_params.get(self.country, tax_params['DE'])
    
    def _load_trades(self):
        """Lädt bestehende Handelsdaten aus der CSV-Datei."""
        if self.trades_file.exists():
            try:
                self.trades = pd.read_csv(self.trades_file).to_dict('records')
                self.logger.info(f"{len(self.trades)} Trades aus {self.trades_file} geladen")
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der Trades: {str(e)}")
                self.trades = []
    
    def _load_positions(self):
        """Lädt bestehende Positionsdaten aus der JSON-Datei."""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, 'r') as f:
                    self.positions = json.load(f)
                self.logger.info(f"Positionen aus {self.positions_file} geladen")
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der Positionen: {str(e)}")
                self.positions = {}
    
    def _save_trades(self):
        """Speichert Handelsdaten in der CSV-Datei."""
        try:
            pd.DataFrame(self.trades).to_csv(self.trades_file, index=False)
            self.logger.debug(f"Trades in {self.trades_file} gespeichert")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Trades: {str(e)}")
    
    def _save_positions(self):
        """Speichert Positionsdaten in der JSON-Datei."""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(self.positions, f, indent=2)
            self.logger.debug(f"Positionen in {self.positions_file} gespeichert")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Positionen: {str(e)}")
    
    def process_trade(self, trade_data: Dict[str, Any]):
        """
        Verarbeitet einen ausgeführten Trade und aktualisiert die Steuerdaten.
        
        Args:
            trade_data: Dictionary mit Handelsdaten
        """
        try:
            # Handels-ID generieren, falls nicht vorhanden
            if 'id' not in trade_data:
                trade_data['id'] = f"trade_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Zeitstempel formatieren
            if 'timestamp' in trade_data and isinstance(trade_data['timestamp'], (int, float)):
                trade_time = datetime.fromtimestamp(trade_data['timestamp'] / 1000.0)
                trade_data['datetime'] = trade_time.isoformat()
            elif 'datetime' not in trade_data:
                trade_data['datetime'] = datetime.now().isoformat()
            
            # Relevante Trade-Daten extrahieren
            trade_id = trade_data.get('id')
            symbol = trade_data.get('symbol')
            side = trade_data.get('side', '').lower()  # 'buy' oder 'sell'
            amount = float(trade_data.get('amount', 0))
            price = float(trade_data.get('price', 0))
            cost = float(trade_data.get('cost', amount * price))
            fee = float(trade_data.get('fee', {}).get('cost', 0))
            fee_currency = trade_data.get('fee', {}).get('currency', 'USDT')
            
            # Überprüfe, ob alle erforderlichen Daten vorhanden sind
            if not all([trade_id, symbol, side, amount, price]):
                self.logger.warning(f"Unvollständige Trade-Daten: {trade_data}")
                return
            
            # Trade-Aufzeichnung erstellen
            trade_record = {
                'id': trade_id,
                'datetime': trade_data.get('datetime'),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'cost': cost,
                'fee': fee,
                'fee_currency': fee_currency,
                'realized_pnl': 0.0  # Wird später aktualisiert, falls es sich um einen Verkauf handelt
            }
            
            # Füge trade_record zu trades hinzu, falls noch nicht vorhanden
            if not any(t.get('id') == trade_id for t in self.trades):
                self.trades.append(trade_record)
                
                # Aktualisiere Positionen basierend auf dem Trade
                if side == 'buy':
                    self._add_to_position(symbol, amount, price, fee, trade_data.get('datetime'))
                elif side == 'sell':
                    pnl, remaining = self._reduce_position(symbol, amount, price, fee, trade_data.get('datetime'))
                    trade_record['realized_pnl'] = pnl
                
                # Speichere Änderungen
                self._save_trades()
                self._save_positions()
                
                self.logger.info(f"Trade {trade_id} verarbeitet: {side} {amount} {symbol} @ {price}")
            else:
                self.logger.debug(f"Trade {trade_id} wurde bereits verarbeitet")
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung des Trades: {str(e)}")
    
    def _add_to_position(self, symbol: str, amount: float, price: float, fee: float, timestamp: str):
        """
        Fügt einen Kauf zur Position hinzu.
        
        Args:
            symbol: Das gehandelte Symbol
            amount: Die gekaufte Menge
            price: Der Kaufpreis pro Einheit
            fee: Die Gebühr für den Handel
            timestamp: Zeitstempel des Kaufs
        """
        # Erstelle Position-Dict, falls das Symbol noch nicht existiert
        if symbol not in self.positions:
            self.positions[symbol] = []
        
        # Positionen für das Symbol holen
        symbol_positions = self.positions[symbol]
        
        # Neue Position hinzufügen
        position = {
            'amount': amount,
            'price': price,
            'fee': fee,
            'cost': price * amount + fee,
            'timestamp': timestamp
        }
        
        symbol_positions.append(position)
        self.positions[symbol] = symbol_positions
    
    def _reduce_position(self, symbol: str, amount: float, price: float, fee: float, timestamp: str) -> Tuple[float, float]:
        """
        Reduziert eine Position durch Verkauf und berechnet den realisierten Gewinn/Verlust.
        
        Args:
            symbol: Das gehandelte Symbol
            amount: Die verkaufte Menge
            price: Der Verkaufspreis pro Einheit
            fee: Die Gebühr für den Handel
            timestamp: Zeitstempel des Verkaufs
            
        Returns:
            Tuple aus (realisierter Gewinn/Verlust, verbleibende Menge)
        """
        if symbol not in self.positions or not self.positions[symbol]:
            self.logger.warning(f"Versuch, nicht vorhandene Position zu reduzieren: {symbol}")
            return 0.0, amount  # Keine Position zum Reduzieren
        
        # Positionen für das Symbol holen
        positions = self.positions[symbol]
        
        # Sortiere Positionen basierend auf der gewählten Methode
        if self.method == TaxMethod.FIFO:
            # Erste hinein, erste heraus: Sortiere nach Kaufzeitpunkt (älteste zuerst)
            positions.sort(key=lambda p: p['timestamp'])
        elif self.method == TaxMethod.LIFO:
            # Letzte hinein, erste heraus: Sortiere nach Kaufzeitpunkt (neueste zuerst)
            positions.sort(key=lambda p: p['timestamp'], reverse=True)
        elif self.method == TaxMethod.HIFO:
            # Höchste hinein, erste heraus: Sortiere nach Kaufpreis (höchste zuerst)
            positions.sort(key=lambda p: p['price'], reverse=True)
        
        remaining_to_sell = amount
        total_cost_basis = 0.0
        total_realized_pnl = 0.0
        positions_to_remove = []
        
        # Gehe durch die Positionen, um die Verkaufsmenge zu decken
        for i, position in enumerate(positions):
            if remaining_to_sell <= 0:
                break
                
            position_amount = position['amount']
            position_price = position['price']
            
            if position_amount <= remaining_to_sell:
                # Die gesamte Position wird verkauft
                amount_sold = position_amount
                positions_to_remove.append(i)
            else:
                # Nur ein Teil der Position wird verkauft
                amount_sold = remaining_to_sell
                # Aktualisiere die verbleibende Menge in der Position
                positions[i]['amount'] -= amount_sold
                positions[i]['cost'] = positions[i]['amount'] * positions[i]['price']
            
            # Kosten dieser verkauften Teilposition
            cost_basis = amount_sold * position_price
            total_cost_basis += cost_basis
            
            # Verkaufserlös für diese Teilposition
            proceeds = amount_sold * price
            
            # Realisierter Gewinn/Verlust für diese Teilposition
            realized_pnl = proceeds - cost_basis
            total_realized_pnl += realized_pnl
            
            remaining_to_sell -= amount_sold
        
        # Entferne vollständig verkaufte Positionen (in umgekehrter Reihenfolge, um Indexprobleme zu vermeiden)
        for i in sorted(positions_to_remove, reverse=True):
            positions.pop(i)
        
        # Aktualisiere die Positionen
        self.positions[symbol] = positions
        
        # Anteilige Gebühr abziehen
        fee_proportion = amount / (amount + remaining_to_sell) if amount + remaining_to_sell > 0 else 1
        adjusted_realized_pnl = total_realized_pnl - (fee * fee_proportion)
        
        # Aktualisiere den Gesamtwert der realisierten Gewinne/Verluste
        self.realized_pnl += adjusted_realized_pnl
        
        return adjusted_realized_pnl, remaining_to_sell
    
    def process_order(self, order_data: Dict[str, Any]):
        """
        Verarbeitet Orderinformationen, hauptsächlich für Logging und Aufzeichnung.
        
        Args:
            order_data: Dictionary mit Orderdaten
        """
        # Extrahiere relevante Informationen für Logging
        order_id = order_data.get('id', 'unknown')
        symbol = order_data.get('symbol', 'unknown')
        side = order_data.get('side', 'unknown')
        status = order_data.get('status', 'unknown')
        
        self.logger.debug(f"Order verarbeitet: {order_id} ({symbol}, {side}, {status})")
        
        # Für geschlossene Orders, die zu Trades führen könnten
        # Hinweis: Die eigentliche Steuerberechnung erfolgt, wenn der Trade verarbeitet wird
    
    def calculate_tax_liability(self, year: int = None) -> Dict[str, Any]:
        """
        Berechnet die Steuerpflicht für ein bestimmtes Jahr.
        
        Args:
            year: Das Jahr, für das die Steuerpflicht berechnet werden soll (Standard: aktuelles Jahr)
            
        Returns:
            Dictionary mit Steuerinformationen
        """
        if year is None:
            year = datetime.now().year
        
        self.logger.info(f"Berechne Steuerpflicht für das Jahr {year}")
        
        # Filtere Trades für das angegebene Jahr
        year_trades = [
            trade for trade in self.trades
            if trade.get('datetime', '').startswith(str(year))
        ]
        
        # Berechne Gesamtgewinn/-verlust für das Jahr
        total_profit = sum(trade.get('realized_pnl', 0) for trade in year_trades)
        
        # Steuerberechnungen basierend auf dem konfigurierten Land
        tax_results = {
            'year': year,
            'total_profit': total_profit,
            'total_trades': len(year_trades),
            'currency': 'EUR' if self.country in ['DE', 'AT', 'FR'] else 'USD'
        }
        
        # Länderspezifische Steuerberechnungen
        if self.country == 'DE':
            # Deutsche Steuerberechnung
            if total_profit <= self.exempt_limit:
                # Unter der Freigrenze
                tax_results.update({
                    'taxable_amount': 0,
                    'tax_rate': 0,
                    'tax_due': 0,
                    'exempt_limit': self.exempt_limit,
                    'is_exempt': True
                })
            else:
                # Über der Freigrenze
                tax_rate = self.tax_parameters.get('tax_rate', 0.25)
                solidarity_rate = self.tax_parameters.get('solidarity_surcharge', 0.055)
                
                capital_gains_tax = total_profit * tax_rate
                solidarity_surcharge = capital_gains_tax * solidarity_rate
                total_tax = capital_gains_tax + solidarity_surcharge
                
                tax_results.update({
                    'taxable_amount': total_profit,
                    'tax_rate': tax_rate,
                    'capital_gains_tax': capital_gains_tax,
                    'solidarity_rate': solidarity_rate,
                    'solidarity_surcharge': solidarity_surcharge,
                    'tax_due': total_tax,
                    'exempt_limit': self.exempt_limit,
                    'is_exempt': False
                })
        
        elif self.country == 'US':
            # US-Steuerberechnung
            # Vereinfachte Darstellung - in Wirklichkeit hängt dies vom Einkommen und Steuersatz ab
            short_term = 0
            long_term = 0
            
            # Berechne kurz- und langfristige Gewinne/Verluste
            for trade in year_trades:
                if trade.get('realized_pnl', 0) != 0:
                    # Berechne Haltedauer
                    trade_date = datetime.fromisoformat(trade.get('datetime', '').split('T')[0])
                    
                    # Hier würde man die Kaufdaten aus den Positionen abrufen
                    # Das ist eine vereinfachte Implementierung
                    if 'holding_period_days' in trade and trade['holding_period_days'] > 365:
                        long_term += trade.get('realized_pnl', 0)
                    else:
                        short_term += trade.get('realized_pnl', 0)
            
            # Steuersätze
            short_term_rate = self.tax_parameters.get('short_term_rate', 0.22)
            long_term_rate = self.tax_parameters.get('long_term_rate', 0.15)
            
            # Steuerberechnung
            short_term_tax = short_term * short_term_rate
            long_term_tax = long_term * long_term_rate
            total_tax = short_term_tax + long_term_tax
            
            tax_results.update({
                'short_term_profit': short_term,
                'long_term_profit': long_term,
                'short_term_rate': short_term_rate,
                'long_term_rate': long_term_rate,
                'short_term_tax': short_term_tax,
                'long_term_tax': long_term_tax,
                'tax_due': total_tax
            })
        
        else:
            # Generische Steuerberechnung für andere Länder
            tax_rate = self.tax_parameters.get('tax_rate', 0.2)
            tax_due = total_profit * tax_rate
            
            tax_results.update({
                'taxable_amount': total_profit,
                'tax_rate': tax_rate,
                'tax_due': tax_due
            })
        
        self.logger.info(f"Steuerberechnung für {year} abgeschlossen: {tax_results.get('tax_due')} {tax_results.get('currency')}")
        
        return tax_results
    
    def generate_tax_report(self, year: int = None, output_format: str = 'json') -> Dict[str, Any]:
        """
        Generiert einen Steuerbericht für ein bestimmtes Jahr.
        
        Args:
            year: Das Jahr, für das der Bericht generiert werden soll (Standard: aktuelles Jahr)
            output_format: Format des Berichts ('json', 'csv', oder 'pdf')
            
        Returns:
            Dictionary mit Informationen zum generierten Bericht
        """
        if year is None:
            year = datetime.now().year
        
        self.logger.info(f"Generiere Steuerbericht für {year} im Format {output_format}")
        
        try:
            # Berechne Steuerpflicht
            tax_data = self.calculate_tax_liability(year)
            
            # Filtere Trades für das angegebene Jahr
            year_trades = [
                trade for trade in self.trades
                if trade.get('datetime', '').startswith(str(year))
            ]
            
            # Erstelle vollständigen Bericht
            report = {
                'tax_summary': tax_data,
                'trades': year_trades,
                'generated_at': datetime.now().isoformat(),
                'year': year,
                'method': self.method.value,
                'country': self.country
            }
            
            # Speichere den Bericht im gewünschten Format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if output_format == 'json':
                output_file = self.reports_path / f"tax_report_{year}_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif output_format == 'csv':
                output_file = self.reports_path / f"tax_report_{year}_{timestamp}.csv"
                
                # Zusammenfassung als separate CSV
                summary_file = self.reports_path / f"tax_summary_{year}_{timestamp}.csv"
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Schlüssel', 'Wert'])
                    for key, value in tax_data.items():
                        writer.writerow([key, value])
                
                # Trades als Haupt-CSV
                pd.DataFrame(year_trades).to_csv(output_file, index=False)
                
                # Beide Dateien in der Rückgabe angeben
                return {
                    'status': 'success',
                    'message': f"Steuerbericht für {year} generiert",
                    'summary_file': str(summary_file),
                    'trades_file': str(output_file)
                }
            
            elif output_format == 'pdf':
                # Hier würde man eine PDF-Generierung implementieren
                # Als Platzhalter geben wir eine Fehlermeldung zurück
                return {
                    'status': 'error',
                    'message': "PDF-Generierung wird noch nicht unterstützt"
                }
            
            self.logger.info(f"Steuerbericht gespeichert als {output_file}")
            
            return {
                'status': 'success',
                'message': f"Steuerbericht für {year} generiert",
                'file': str(output_file)
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Generierung des Steuerberichts: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def get_unrealized_pnl(self) -> Dict[str, Any]:
        """
        Berechnet den nicht realisierten Gewinn/Verlust für alle offenen Positionen.
        
        Returns:
            Dictionary mit nicht realisierten Gewinnen/Verlusten pro Symbol und Gesamtwert
        """
        unrealized = {}
        total_unrealized = 0.0
        
        for symbol, positions in self.positions.items():
            if not positions:
                continue
            
            # Hier würde man den aktuellen Marktpreis abrufen
            # In einer echten Implementierung würde man die Datenpipeline verwenden
            market_price = 0.0
            
            # Alternative: Wenn wir keinen aktuellen Preis haben, verwenden wir den letzten bekannten Preis
            for trade in reversed(self.trades):
                if trade.get('symbol') == symbol:
                    market_price = trade.get('price', 0.0)
                    break
            
            # Berechne den nicht realisierten Gewinn/Verlust für dieses Symbol
            symbol_unrealized = 0.0
            total_amount = 0.0
            avg_price = 0.0
            
            for position in positions:
                amount = position.get('amount', 0.0)
                price = position.get('price', 0.0)
                cost = amount * price
                
                total_amount += amount
                
                # Aktualisierte Bewertung der Position
                current_value = amount * market_price
                
                # Nicht realisierter Gewinn/Verlust
                position_unrealized = current_value - cost
                symbol_unrealized += position_unrealized
            
            # Durchschnittspreis berechnen
            if total_amount > 0:
                avg_price = sum(pos.get('amount', 0) * pos.get('price', 0) for pos in positions) / total_amount
            
            unrealized[symbol] = {
                'amount': total_amount,
                'avg_price': avg_price,
                'market_price': market_price,
                'unrealized_pnl': symbol_unrealized
            }
            
            total_unrealized += symbol_unrealized
        
        return {
            'per_symbol': unrealized,
            'total_unrealized': total_unrealized
        }
    
    def get_tax_summary(self, year: int = None) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der Steuerinformationen zurück.
        
        Args:
            year: Das Jahr, für das die Zusammenfassung erstellt werden soll (Standard: aktuelles Jahr)
            
        Returns:
            Dictionary mit Steuer-Zusammenfassung
        """
        if year is None:
            year = datetime.now().year
        
        # Steuerpflicht berechnen
        tax_data = self.calculate_tax_liability(year)
        
        # Nicht realisierten Gewinn/Verlust abrufen
        unrealized_data = self.get_unrealized_pnl()
        
        # Gesamtperformance
        realized_pnl = sum(trade.get('realized_pnl', 0) for trade in self.trades 
                           if trade.get('datetime', '').startswith(str(year)))
        
        # Erweiterte Zusammenfassung erstellen
        summary = {
            'year': year,
            'total_trades': len([t for t in self.trades if t.get('datetime', '').startswith(str(year))]),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_data.get('total_unrealized', 0),
            'total_pnl': realized_pnl + unrealized_data.get('total_unrealized', 0),
            'tax_due': tax_data.get('tax_due', 0),
            'method': self.method.value,
            'country': self.country,
            'exempt_limit': self.exempt_limit
        }
        
        return summary
    
    def set_tax_method(self, method: Union[str, TaxMethod]) -> bool:
        """
        Ändert die verwendete Steuermethode.
        
        Args:
            method: Neue Steuermethode (FIFO, LIFO, HIFO)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if isinstance(method, str):
                # Konvertieren von Zeichenkette zu Enum
                if method.upper() in [m.name for m in TaxMethod]:
                    self.method = TaxMethod[method.upper()]
                else:
                    # Versuche die Wert-Konvertierung
                    for m in TaxMethod:
                        if m.value == method.lower():
                            self.method = m
                            break
                    else:
                        self.logger.error(f"Ungültige Steuermethode: {method}")
                        return False
            elif isinstance(method, TaxMethod):
                self.method = method
            else:
                self.logger.error(f"Ungültiger Typ für Steuermethode: {type(method)}")
                return False
            
            self.logger.info(f"Steuermethode auf {self.method.name} ({self.method.value}) geändert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Ändern der Steuermethode: {str(e)}")
            return False
    
    def set_country(self, country_code: str) -> bool:
        """
        Ändert das konfigurierte Land für Steuerberechnungen.
        
        Args:
            country_code: Ländercode (z.B. 'DE', 'US', 'UK')
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            country_code = country_code.upper()
            
            # Prüfe, ob das Land unterstützt wird
            supported_countries = ['DE', 'US', 'UK', 'AT', 'CH', 'FR']
            
            if country_code not in supported_countries:
                self.logger.warning(f"Land {country_code} wird möglicherweise nicht vollständig unterstützt")
            
            self.country = country_code
            
            # Steuerparameter aktualisieren
            self.tax_parameters = self._get_tax_parameters()
            
            self.logger.info(f"Land für Steuerberechnungen auf {self.country} geändert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Ändern des Landes: {str(e)}")
            return False
    
    def import_trades(self, file_path: str, format: str = 'csv') -> Dict[str, Any]:
        """
        Importiert Handelsdaten aus einer externen Datei.
        
        Args:
            file_path: Pfad zur Import-Datei
            format: Format der Datei ('csv' oder 'json')
            
        Returns:
            Ergebnis des Imports als Dictionary
        """
        try:
            self.logger.info(f"Importiere Handelsdaten aus {file_path} im Format {format}")
            
            imported_trades = []
            
            if format.lower() == 'csv':
                imported_trades = pd.read_csv(file_path).to_dict('records')
            elif format.lower() == 'json':
                with open(file_path, 'r') as f:
                    imported_trades = json.load(f)
            else:
                return {
                    'status': 'error',
                    'message': f"Nicht unterstütztes Format: {format}"
                }
            
            # Zähle vorhandene und neue Trades
            existing_ids = {trade.get('id') for trade in self.trades}
            new_trades = [trade for trade in imported_trades if trade.get('id') not in existing_ids]
            
            # Verarbeite neue Trades
            for trade in new_trades:
                self.process_trade(trade)
            
            self.logger.info(f"{len(new_trades)} neue Trades importiert")
            
            return {
                'status': 'success',
                'message': f"{len(new_trades)} neue Trades importiert",
                'total_imported': len(imported_trades),
                'new_trades': len(new_trades),
                'already_existing': len(imported_trades) - len(new_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Import von Handelsdaten: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def export_trades(self, output_file: str = None, format: str = 'csv', year: int = None) -> Dict[str, Any]:
        """
        Exportiert Handelsdaten in eine externe Datei.
        
        Args:
            output_file: Pfad zur Export-Datei (optional)
            format: Format der Datei ('csv' oder 'json')
            year: Exportiert nur Trades aus diesem Jahr (optional)
            
        Returns:
            Ergebnis des Exports als Dictionary
        """
        try:
            # Filtere Trades nach Jahr, falls angegeben
            filtered_trades = self.trades
            if year is not None:
                filtered_trades = [
                    trade for trade in self.trades
                    if trade.get('datetime', '').startswith(str(year))
                ]
            
            # Erstelle Exportdateiname, falls nicht angegeben
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                year_suffix = f"_{year}" if year is not None else ""
                output_file = str(self.data_path / f"trades_export{year_suffix}_{timestamp}.{format}")
            
            # Exportiere im angegebenen Format
            if format.lower() == 'csv':
                pd.DataFrame(filtered_trades).to_csv(output_file, index=False)
            elif format.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(filtered_trades, f, indent=2, default=str)
            else:
                return {
                    'status': 'error',
                    'message': f"Nicht unterstütztes Format: {format}"
                }
            
            self.logger.info(f"{len(filtered_trades)} Trades nach {output_file} exportiert")
            
            return {
                'status': 'success',
                'message': f"{len(filtered_trades)} Trades exportiert",
                'file': output_file,
                'format': format
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Export von Handelsdaten: {str(e)}")
            return {
                'status': 'error',
                'message': f"Fehler: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über Trades und steuerrelevante Informationen zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        current_year = datetime.now().year
        
        # Zähle Trades pro Jahr
        trades_by_year = {}
        for trade in self.trades:
            trade_date = trade.get('datetime', '')
            if trade_date:
                year = trade_date[:4]  # Extrahiere Jahr aus dem ISO-Datum
                trades_by_year[year] = trades_by_year.get(year, 0) + 1
        
        # Berechne Gewinn/Verlust pro Jahr
        pnl_by_year = {}
        for trade in self.trades:
            trade_date = trade.get('datetime', '')
            if trade_date:
                year = trade_date[:4]
                pnl_by_year[year] = pnl_by_year.get(year, 0) + trade.get('realized_pnl', 0)
        
        # Ermittle die am häufigsten gehandelten Symbole
        symbol_counts = {}
        for trade in self.trades:
            symbol = trade.get('symbol', 'unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Berechne Performance-Metriken
        total_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if trade.get('realized_pnl', 0) > 0)
        unprofitable_trades = sum(1 for trade in self.trades if trade.get('realized_pnl', 0) < 0)
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Steuerprognose für das aktuelle Jahr
        current_year_tax = self.calculate_tax_liability(current_year)
        
        # Erstelle Statistik-Dictionary
        stats = {
            'total_trades': total_trades,
            'trades_by_year': trades_by_year,
            'pnl_by_year': pnl_by_year,
            'total_realized_pnl': sum(trade.get('realized_pnl', 0) for trade in self.trades),
            'current_year_pnl': pnl_by_year.get(str(current_year), 0),
            'profitable_trades': profitable_trades,
            'unprofitable_trades': unprofitable_trades,
            'win_rate': win_rate,
            'top_symbols': [{"symbol": s, "count": c} for s, c in top_symbols],
            'open_positions': len(self.positions),
            'tax_method': self.method.value,
            'country': self.country,
            'current_year_tax_due': current_year_tax.get('tax_due', 0)
        }
        
        return stats

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'default_method': 'FIFO',
        'country': 'DE',
        'exempt_limit': 600,
        'data_path': 'data/tax'
    }
    
    # TaxModule initialisieren
    tax_module = TaxModule(config)
    
    # Beispiel-Trade verarbeiten
    example_trade = {
        'id': 'trade_example1',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 0.1,
        'price': 50000,
        'cost': 5000,
        'fee': 5,
        'fee_currency': 'USDT',
        'datetime': '2025-01-15T12:00:00'
    }
    
    tax_module.process_trade(example_trade)
    
    # Steuerzusammenfassung abrufen
    summary = tax_module.get_tax_summary(2025)
    print(f"Steuerzusammenfassung: {summary}")

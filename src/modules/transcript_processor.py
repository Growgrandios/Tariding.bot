# transcript_processor.py

import os
import re
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Union

# Gemma Integration
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/transcript_processor.log"),
        logging.StreamHandler()
    ]
)

class TranscriptProcessor:
    """
    Verarbeitet YouTube-Transkripte und extrahiert Trading-Wissen
    mit Hilfe von Gemma 3.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialisiert den TranscriptProcessor.
        
        Args:
            config: Konfigurationseinstellungen
        """
        self.logger = logging.getLogger("TranscriptProcessor")
        self.logger.info("Initialisiere TranscriptProcessor...")
        
        # Konfiguration laden
        self.config = config or {}
        self.model_path = self.config.get('model_path', "google/gemma-3-8b")
        self.device = self.config.get('device', "auto")
        
        # Pfade für Daten
        self.base_path = Path(self.config.get('data_path', "data"))
        self.transcript_path = self.base_path / "transcripts"
        self.knowledge_path = self.base_path / "knowledge"
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.transcript_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Lade Gemma 3 Modell
        self.logger.info(f"Lade Gemma 3 Modell von {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Für größere Modelle optionale 4-bit Quantisierung
            if "27b" in self.model_path or "30b" in self.model_path:
                self.logger.info("Verwende 4-bit Quantisierung für großes Modell")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    load_in_4bit=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16
                )
            self.logger.info("Gemma 3 Modell erfolgreich geladen")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
        # Kategorien für das strukturierte Wissen
        self.categories = [
            "trading_strategies", 
            "technical_indicators",
            "risk_management",
            "market_analysis",
            "trading_psychology",
            "entry_exit_rules",
            "position_sizing"
        ]
        
        self.logger.info("TranscriptProcessor wurde erfolgreich initialisiert")
    
    def load_transcript(self, file_path: Union[str, Path]) -> str:
        """
        Lädt ein Transkript aus einer Datei.
        
        Args:
            file_path: Pfad zur Transkriptdatei
            
        Returns:
            Der Inhalt des Transkripts als String
        """
        file_path = Path(file_path)
        self.logger.info(f"Lade Transkript: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.info(f"Transkript geladen: {len(content)} Zeichen")
            return content
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Transkripts {file_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Erkennt die Sprache des Transkripts.
        
        Args:
            text: Der zu analysierende Text
            
        Returns:
            Sprachcode ('en' oder 'de')
        """
        # Einfache Heuristik: Betrachte typische deutsche Wörter
        german_words = ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'auf', 'nicht', 'auch', 'eine']
        
        # Zähle, wie oft deutsche Wörter im Text vorkommen (case-insensitive)
        text_lower = text.lower()
        german_count = sum(1 for word in german_words if f' {word} ' in text_lower)
        
        # Wenn mehr als 5 typisch deutsche Wörter gefunden wurden, ist es wahrscheinlich Deutsch
        if german_count > 5:
            self.logger.info("Sprache erkannt: Deutsch")
            return 'de'
        else:
            self.logger.info("Sprache erkannt: Englisch")
            return 'en'
    
    def clean_transcript(self, text: str) -> str:
        """
        Bereinigt das Transkript von unnötigen Elementen.
        
        Args:
            text: Das zu bereinigende Transkript
            
        Returns:
            Bereinigtes Transkript
        """
        self.logger.info("Bereinige Transkript...")
        
        # Entferne Zeitstempel (typisch für YouTube-Transkripte)
        text = re.sub(r'\[\d+:\d+\]', '', text)
        
        # Entferne mehrfache Leerzeichen
        text = re.sub(r' +', ' ', text)
        
        # Entferne mehrfache Zeilenumbrüche
        text = re.sub(r'\n+', '\n', text)
        
        self.logger.info("Transkript bereinigt")
        return text.strip()
    
    def split_transcript(self, text: str, max_chunk_size: int = 15000) -> List[str]:
        """
        Teilt das Transkript in kleinere Abschnitte auf, die vom Modell verarbeitet werden können.
        
        Args:
            text: Das Transkript
            max_chunk_size: Maximale Zeichenanzahl pro Chunk
            
        Returns:
            Liste von Textabschnitten
        """
        self.logger.info(f"Teile Transkript in Abschnitte (max {max_chunk_size} Zeichen)...")
        
        # Teile am Absatz
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Wenn der aktuelle Chunk plus neuer Absatz nicht zu groß wird
            if len(current_chunk) + len(paragraph) < max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Aktuellen Chunk speichern und neuen beginnen
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Letzten Chunk hinzufügen, falls vorhanden
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        self.logger.info(f"Transkript in {len(chunks)} Abschnitte geteilt")
        return chunks
    
    def extract_knowledge_from_chunk(self, chunk: str, language: str) -> Dict[str, List[str]]:
        """
        Extrahiert strukturiertes Wissen aus einem Transkriptabschnitt.
        
        Args:
            chunk: Ein Abschnitt des Transkripts
            language: Sprachcode ('en' oder 'de')
            
        Returns:
            Wörterbuch mit extrahiertem Wissen nach Kategorien
        """
        self.logger.info(f"Extrahiere Wissen aus Transkriptabschnitt ({len(chunk)} Zeichen)...")
        
        # Prompt für das Modell basierend auf der erkannten Sprache
        if language == 'de':
            prompt = f"""
Du bist ein Finanzexperte, der Transkripte von Trading-Videos analysiert.

Extrahiere wichtiges Trading-Wissen aus dem folgenden Transkript und strukturiere es in diese Kategorien:
- Trading-Strategien: Spezifische Handelsmethoden, Setups und Marktansätze
- Technische Indikatoren: Verwendung von technischen Indikatoren und Chart-Mustern
- Risikomanagement: Money Management, Position Sizing und Risikokontrolle
- Marktanalyse: Marktphasen, Volumenanalyse und Marktstruktur
- Trading-Psychologie: Emotionale Kontrolle und psychologische Aspekte des Tradings
- Entry-Exit-Regeln: Spezifische Regeln für Einstiegs- und Ausstiegspunkte
- Position Sizing: Methoden zur Bestimmung der optimalen Positionsgröße

Ignoriere irrelevante Informationen. Gib nur klare, präzise und umsetzbare Informationen zurück.
Für jede Kategorie, gib die Informationen als Aufzählungspunkte mit je einer konkreten Trading-Regel oder Information pro Zeile.
Wenn in einer Kategorie keine relevanten Informationen enthalten sind, gib "Keine relevanten Informationen" zurück.

Transkript:
{chunk}

Strukturiertes Wissen (in deutscher Sprache):
"""
        else:  # Englisch
            prompt = f"""
You are a finance expert analyzing transcripts from trading videos.

Extract important trading knowledge from the following transcript and structure it into these categories:
- Trading Strategies: Specific trading methods, setups, and market approaches
- Technical Indicators: Use of technical indicators and chart patterns
- Risk Management: Money management, position sizing, and risk control
- Market Analysis: Market phases, volume analysis, and market structure
- Trading Psychology: Emotional control and psychological aspects of trading
- Entry-Exit Rules: Specific rules for entry and exit points
- Position Sizing: Methods for determining optimal position size

Ignore irrelevant information. Only return clear, precise, and actionable information.
For each category, provide the information as bullet points with one specific trading rule or piece of information per line.
If a category contains no relevant information, return "No relevant information".

Transcript:
{chunk}

Structured Knowledge (in English):
"""
        
        try:
            # Tokenisiere den Prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generiere die Antwort mit kontrollierter Temperatur
            output = self.model.generate(
                inputs.input_ids,
                max_new_tokens=4000,  # Genug für eine ausführliche Antwort
                do_sample=True,
                temperature=0.2,  # Niedrigere Temperatur für fokussiertere Antworten
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            # Dekodiere die Antwort
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extrahiere den relevanten Teil nach dem Prompt
            if language == 'de':
                response = response.split("Strukturiertes Wissen (in deutscher Sprache):")[-1].strip()
            else:
                response = response.split("Structured Knowledge (in English):")[-1].strip()
            
            # Parsen der Antwort in ein strukturiertes Format
            result = self._parse_model_response(response, language)
            
            self.logger.info("Wissensextraktion abgeschlossen")
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Wissensextraktion: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {category: [] for category in self.categories}
    
    def _parse_model_response(self, response: str, language: str) -> Dict[str, List[str]]:
        """
        Parst die Antwort des Modells in ein strukturiertes Format.
        
        Args:
            response: Die Antwort des Modells
            language: Sprachcode ('en' oder 'de')
            
        Returns:
            Strukturiertes Wissen nach Kategorien
        """
        result = {category: [] for category in self.categories}
        
        # Mapping von lokalisierten Kategorienamen zu internen Schlüsseln
        category_mapping = {
            # Deutsch
            'trading-strategien': 'trading_strategies',
            'technische indikatoren': 'technical_indicators',
            'risikomanagement': 'risk_management',
            'marktanalyse': 'market_analysis',
            'trading-psychologie': 'trading_psychology',
            'entry-exit-regeln': 'entry_exit_rules',
            'position sizing': 'position_sizing',
            
            # Englisch
            'trading strategies': 'trading_strategies',
            'technical indicators': 'technical_indicators',
            'risk management': 'risk_management',
            'market analysis': 'market_analysis',
            'trading psychology': 'trading_psychology',
            'entry-exit rules': 'entry_exit_rules',
            'position sizing': 'position_sizing'
        }
        
        current_category = None
        
        # Zeile für Zeile die Antwort durchgehen
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Prüfe, ob dies eine Kategorieüberschrift ist
            is_header = False
            for category_name, internal_key in category_mapping.items():
                if line.lower().startswith(category_name) or f"- {category_name}:" in line.lower():
                    current_category = internal_key
                    is_header = True
                    break
            
            if is_header:
                continue
                
            # Wenn wir in einer Kategorie sind und die Zeile ist kein "keine relevanten Informationen"
            if current_category and not any(phrase in line.lower() for phrase in 
                                          ["keine relevanten informationen", "no relevant information"]):
                # Listenpunkt bereinigen
                if line.startswith('- '):
                    line = line[2:].strip()
                elif line.startswith('* '):
                    line = line[2:].strip()
                    
                # Füge den Punkt zur entsprechenden Kategorie hinzu
                result[current_category].append(line)
        
        return result
    
    def process_transcript(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Verarbeitet eine vollständige Transkriptdatei und extrahiert strukturiertes Wissen.
        
        Args:
            file_path: Pfad zur Transkriptdatei
            
        Returns:
            Dictionary mit Ergebnisinformationen
        """
        try:
            self.logger.info(f"Starte Verarbeitung der Transkriptdatei: {file_path}")
            
            # Lade und bereinige das Transkript
            transcript = self.load_transcript(file_path)
            cleaned_transcript = self.clean_transcript(transcript)
            
            # Erkenne die Sprache
            language = self.detect_language(cleaned_transcript)
            
            # Teile das Transkript in Abschnitte
            chunks = self.split_transcript(cleaned_transcript)
            
            # Extrahiere Wissen aus jedem Abschnitt
            all_knowledge = {category: [] for category in self.categories}
            
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Verarbeite Chunk {i+1}/{len(chunks)}...")
                chunk_knowledge = self.extract_knowledge_from_chunk(chunk, language)
                
                # Kombiniere die Ergebnisse
                for category in self.categories:
                    all_knowledge[category].extend(chunk_knowledge[category])
            
            # Entferne Duplikate
            for category in self.categories:
                all_knowledge[category] = list(set(all_knowledge[category]))
            
            # Speichere das extrahierte Wissen
            file_path = Path(file_path)
            output_file = self.knowledge_path / f"{file_path.stem}_knowledge.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Wissen aus Transkript extrahiert und gespeichert in: {output_file}")
            
            # Erstelle Ergebniszusammenfassung
            result = {
                'status': 'success',
                'file': str(file_path),
                'language': language,
                'chunks': len(chunks),
                'output_file': str(output_file),
                'knowledge_items': {category: len(items) for category, items in all_knowledge.items()},
                'total_items': sum(len(items) for items in all_knowledge.values())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'file': str(file_path) if 'file_path' in locals() else None,
                'error': str(e)
            }
    
    def process_directory(self, directory: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """
        Verarbeitet alle Transkriptdateien in einem Verzeichnis.
        
        Args:
            directory: Pfad zum Verzeichnis (Standard: self.transcript_path)
            
        Returns:
            Liste mit Ergebnissen für jede verarbeitete Datei
        """
        directory = Path(directory) if directory else self.transcript_path
        self.logger.info(f"Verarbeite alle Transkripte im Verzeichnis: {directory}")
        
        results = []
        
        try:
            # Finde alle .txt-Dateien im Verzeichnis
            transcript_files = list(directory.glob("*.txt"))
            self.logger.info(f"{len(transcript_files)} Transkript-Dateien gefunden")
            
            for file_path in transcript_files:
                self.logger.info(f"Verarbeite Datei: {file_path}")
                result = self.process_transcript(file_path)
                results.append(result)
                
                # Kurze Pause zwischen Dateien, um das Modell nicht zu überlasten
                time.sleep(1)
            
            # Zusammenfassung
            success_count = sum(1 for r in results if r.get('status') == 'success')
            self.logger.info(f"Verarbeitung abgeschlossen: {success_count} von {len(results)} erfolgreich")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verzeichnis-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            return results
    
    def combine_knowledge_files(self, output_file: str = "combined_knowledge.json") -> Dict[str, Any]:
        """
        Kombiniert alle extrahierten Wissensdateien zu einer umfassenden Wissensbasis.
        
        Args:
            output_file: Name der Ausgabedatei
            
        Returns:
            Dictionary mit Ergebnisinformationen
        """
        self.logger.info("Kombiniere alle Wissensdateien...")
        
        try:
            # Finde alle JSON-Dateien im knowledge-Verzeichnis
            knowledge_files = list(self.knowledge_path.glob("*_knowledge.json"))
            self.logger.info(f"{len(knowledge_files)} Wissensdateien gefunden")
            
            if not knowledge_files:
                return {
                    'status': 'error',
                    'message': 'Keine Wissensdateien gefunden'
                }
            
            # Kombinierte Wissensbasis initialisieren
            combined_knowledge = {category: [] for category in self.categories}
            
            # Alle Dateien durchgehen und Wissen kombinieren
            for file_path in knowledge_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    knowledge = json.load(f)
                
                # Kategorien kombinieren
                for category in self.categories:
                    if category in knowledge:
                        combined_knowledge[category].extend(knowledge[category])
            
            # Duplikate entfernen
            for category in self.categories:
                combined_knowledge[category] = list(set(combined_knowledge[category]))
                
                # Nach Länge sortieren (kürzere Einträge zuerst)
                combined_knowledge[category].sort(key=len)
            
            # Kombinierte Wissensbasis speichern
            output_path = self.knowledge_path / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_knowledge, f, ensure_ascii=False, indent=4)
            
            # Statistiken erstellen
            stats = {
                'total_files': len(knowledge_files),
                'total_items': sum(len(items) for items in combined_knowledge.values()),
                'items_per_category': {category: len(items) for category, items in combined_knowledge.items()}
            }
            
            self.logger.info(f"Kombinierte Wissensbasis gespeichert in: {output_path}")
            self.logger.info(f"Gesamtzahl der Wissenseinträge: {stats['total_items']}")
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kombinieren der Wissensdateien: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_trading_rules(self, output_file: str = "trading_rules.json") -> Dict[str, Any]:
        """
        Generiert konkrete Trading-Regeln basierend auf der kombinierten Wissensbasis.
        
        Args:
            output_file: Name der Ausgabedatei
            
        Returns:
            Dictionary mit Ergebnisinformationen
        """
        self.logger.info("Generiere Trading-Regeln aus der Wissensbasis...")
        
        try:
            # Kombinierte Wissensbasis laden
            combined_file = self.knowledge_path / "combined_knowledge.json"
            
            if not combined_file.exists():
                # Erstelle die kombinierte Wissensbasis, falls sie noch nicht existiert
                self.combine_knowledge_files()
                
                if not combined_file.exists():
                    return {
                        'status': 'error',
                        'message': 'Kombinierte Wissensbasis konnte nicht erstellt werden'
                    }
            
            with open(combined_file, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
            
            # Prompt erstellen
            knowledge_text = ""
            for category in self.categories:
                if category in knowledge and knowledge[category]:
                    category_name = category.replace('_', ' ').title()
                    knowledge_text += f"\n{category_name}:\n"
                    for item in knowledge[category][:20]:  # Begrenze auf 20 Einträge pro Kategorie
                        knowledge_text += f"- {item}\n"
            
            prompt = f"""
Als Experte für algorithmisches Trading, verwende die folgende Wissensbasis, um ein Set von konkreten, 
algorithmisch umsetzbaren Trading-Regeln zu erstellen. Die Regeln sollten präzise und direkt umsetzbar sein.

Wissensbasis:
{knowledge_text}

Bitte erstelle Trading-Regeln für folgende Kategorien:
1. Entry Rules: Präzise Regeln, wann eine Position eröffnet werden sollte
2. Exit Rules: Klare Regeln, wann eine Position geschlossen werden sollte
3. Risk Management: Konkrete Regeln zur Risikobegrenzung
4. Position Sizing: Formel oder Methode zur Berechnung der Positionsgröße

Die Regeln sollten:
- Algorithmisch umsetzbar sein (mit klaren numerischen Werten und Bedingungen)
- Auf technischen Indikatoren oder Preisaktionen basieren
- Konsistent und nicht widersprüchlich sein
- Eine vollständige Trading-Strategie darstellen

Formatiere jede Regel als einzelne, präzise Aussage.
"""
            
            # Generiere Trading-Regeln mit dem Modell
            self.logger.info("Generiere Trading-Regeln mit Gemma 3...")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                inputs.input_ids,
                max_new_tokens=3000,
                do_sample=True,
                temperature=0.1,  # Niedrigere Temperatur für präzisere Regeln
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extrahiere den relevanten Teil (nach dem Prompt)
            response = response.split(prompt)[-1].strip()
            
            # Parsen der generierten Regeln
            rules = self._parse_generated_rules(response)
            
            # Speichere die Trading-Regeln
            output_path = self.knowledge_path / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Trading-Regeln gespeichert in: {output_path}")
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'rules': rules
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Generierung von Trading-Regeln: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _parse_generated_rules(self, text: str) -> Dict[str, List[str]]:
        """
        Parst die vom Modell generierten Trading-Regeln.
        
        Args:
            text: Die generierte Antwort
            
        Returns:
            Dictionary mit kategorisierten Trading-Regeln
        """
        categories = {
            'entry_rules': [],
            'exit_rules': [],
            'risk_management': [],
            'position_sizing': []
        }
        
        # Mapping für verschiedene Kategorie-Überschriften
        category_mapping = {
            'entry rules': 'entry_rules',
            'exit rules': 'exit_rules',
            'risk management': 'risk_management',
            'position sizing': 'position_sizing'
        }
        
        current_category = None
        
        # Zeile für Zeile durchgehen
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Prüfe, ob dies eine Kategorieüberschrift ist
            is_header = False
            for header, key in category_mapping.items():
                if re.search(rf'^\d*\.?\s*{header}', line.lower()):
                    current_category = key
                    is_header = True
                    break
            
            if is_header:
                continue
            
            # Wenn wir in einer Kategorie sind und die Zeile ein Regelpunkt ist
            if current_category and (line.startswith('-') or re.match(r'^\d+\.', line)):
                # Listenpunkt bereinigen
                if line.startswith('-'):
                    line = line[1:].strip()
                else:
                    # Entferne Nummerierung (z.B. "1. ")
                    line = re.sub(r'^\d+\.\s*', '', line)
                
                # Regel zur entsprechenden Kategorie hinzufügen
                categories[current_category].append(line)
        
        return categories

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'model_path': "google/gemma-3-8b",
        'device': "auto",
        'data_path': "data"
    }
    
    # TranscriptProcessor initialisieren
    processor = TranscriptProcessor(config)
    
    # Beispiel für die Verarbeitung eines Transkripts
    # processor.process_transcript("data/transcripts/example_transcript.txt")
    
    # Beispiel für die Verarbeitung aller Transkripte im Verzeichnis
    # processor.process_directory()
    
    # Beispiel für die Kombination aller Wissensdateien
    # processor.combine_knowledge_files()
    
    # Beispiel für die Generierung von Trading-Regeln
    # processor.generate_trading_rules()

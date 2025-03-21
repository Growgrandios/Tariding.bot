# transcript_processor.py

import os
import re
import json
import logging
import time
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Für Sprachmodelle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class TranscriptProcessor:
    """
    Verarbeitet YouTube-Transkripte und extrahiert Trading-Wissen
    mit Hilfe von Large Language Models (LLMs).
    
    Integriert sich nahtlos in die bestehende Bot-Architektur und
    liefert strukturierte Trading-Erkenntnisse aus Video-Transkripten.
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
        
        # Modellkonfiguration
        self.model_path = self.config.get('model_path', "google/gemma-2-9b")
        self.device = self.config.get('device', "auto")
        self.device = self._resolve_device(self.device)
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.2)
        self.chunk_size = self.config.get('chunk_size', 15000)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 2)
        
        # Hugging Face Token aus Konfiguration oder Umgebungsvariable
        self.hf_token = self.config.get('hf_token', os.environ.get("HUGGINGFACE_TOKEN", ""))
        
        # Pfade für Daten
        self.base_path = Path(self.config.get('data_path', "data"))
        self.transcript_path = Path(self.config.get('transcript_path', self.base_path / "transcripts"))
        self.knowledge_path = Path(self.config.get('knowledge_path', self.base_path / "knowledge"))
        
        # Stelle sicher, dass die Verzeichnisse existieren
        self.transcript_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Kategorien für das strukturierte Wissen
        self.categories = self.config.get('categories', [
            "trading_strategies",
            "technical_indicators",
            "risk_management",
            "market_analysis",
            "trading_psychology",
            "entry_exit_rules",
            "position_sizing"
        ])
        
        # Modell und Tokenizer (lazy loading)
        self.tokenizer = None
        self.model = None
        self._model_lock = threading.Lock()
        self._hf_auth_done = False
        
        # Fortschritts-Callback
        self.progress_callback = None
        
        # Cache für zwischengespeicherte Ergebnisse
        self.results_cache = {}
        
        self.logger.info("TranscriptProcessor erfolgreich initialisiert")
    
    def _resolve_device(self, device_setting: str) -> str:
        """
        Ermittelt das zu verwendende Gerät (CPU/GPU).
        
        Args:
            device_setting: Einstellung ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            Gerätestring
        """
        if device_setting == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_setting
    
    def register_progress_callback(self, callback: Callable[[str, float, Dict[str, Any]], None]):
        """
        Registriert einen Callback für Fortschrittsaktualisierungen.
        
        Args:
            callback: Funktion, die bei Fortschrittsänderungen aufgerufen wird.
                     Parameter: (task_id, progress_percentage, additional_info)
        """
        self.progress_callback = callback
        self.logger.debug("Fortschritts-Callback registriert")
    
    def _report_progress(self, task_id: str, progress: float, info: Dict[str, Any] = None):
        """
        Meldet Fortschrittsaktualisierungen an den registrierten Callback.
        
        Args:
            task_id: Eindeutige Kennung der Aufgabe
            progress: Fortschritt als Prozentwert (0.0 bis 1.0)
            info: Zusätzliche Informationen
        """
        if self.progress_callback:
            try:
                self.progress_callback(task_id, progress, info or {})
            except Exception as e:
                self.logger.error(f"Fehler im Fortschritts-Callback: {str(e)}")
    
    def _authenticate_huggingface(self) -> bool:
        """
        Authentifiziert bei Hugging Face Hub.
        
        Returns:
            True bei erfolgreicher Authentifizierung, sonst False
        """
        if self._hf_auth_done:
            return True
            
        if not self.hf_token:
            self.logger.warning("Kein Hugging Face Token gefunden. Setze HF_TOKEN in der Umgebung oder Konfiguration.")
            return False
            
        try:
            # Authentifizierung mit dem Token
            login(token=self.hf_token, add_to_git_credential=False)
            self._hf_auth_done = True
            self.logger.info("Hugging Face Authentifizierung erfolgreich")
            return True
        except Exception as e:
            self.logger.error(f"Hugging Face Authentifizierung fehlgeschlagen: {str(e)}")
            return False
    
    def _load_model(self) -> bool:
        """
        Lädt das Modell und den Tokenizer (lazy loading mit Thread-Sicherheit).
        
        Returns:
            True bei erfolgreicher Initialisierung, sonst False
        """
        # Thread-sichere Überprüfung, ob Modell bereits geladen ist
        if self.tokenizer is not None and self.model is not None:
            return True
            
        # Authentifizierung bei Hugging Face
        if not self._authenticate_huggingface():
            return False
            
        # Thread-sichere Modellinitialisierung
        with self._model_lock:
            # Erneute Überprüfung nach Erhalten des Locks
            if self.tokenizer is not None and self.model is not None:
                return True
                
            self.logger.info(f"Lade Modell {self.model_path} auf {self.device}...")
            
            # Wiederholungslogik
            max_versuche = 3
            wartezeit = 5  # Sekunden
            
            for versuch in range(max_versuche):
                try:
                    # Tokenizer laden
                    self.logger.debug("Lade Tokenizer...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        token=self.hf_token,
                        use_auth_token=True
                    )
                    
                    # Modell laden
                    self.logger.debug("Lade Modell...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map=self.device,
                        torch_dtype=torch.float16,  # Speichereffizienter
                        token=self.hf_token,
                        use_auth_token=True
                    )
                    
                    self.logger.info(f"Modell erfolgreich geladen auf {self.device}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Fehler beim Laden des Modells (Versuch {versuch+1}/{max_versuche}): {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    if versuch < max_versuche - 1:
                        self.logger.info(f"Neuer Versuch in {wartezeit} Sekunden...")
                        time.sleep(wartezeit)
                        wartezeit *= 2  # Exponentielle Wartezeit
                    else:
                        self.logger.critical("Modell konnte nach mehreren Versuchen nicht geladen werden")
                        return False
    
    def load_transcript(self, file_path: Union[str, Path]) -> str:
        """
        Lädt ein Transkript aus einer Datei.
        
        Args:
            file_path: Pfad zur Transkriptdatei
            
        Returns:
            Der Inhalt des Transkripts als String
            
        Raises:
            FileNotFoundError: Wenn die Datei nicht gefunden wird
            UnicodeDecodeError: Wenn die Datei nicht als UTF-8 dekodiert werden kann
        """
        file_path = Path(file_path)
        self.logger.info(f"Lade Transkript: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.logger.info(f"Transkript geladen: {len(content)} Zeichen")
            return content
            
        except UnicodeDecodeError:
            # Versuche andere Codierungen, falls UTF-8 fehlschlägt
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                self.logger.warning(f"Transkript mit Latin-1 Codierung geladen: {len(content)} Zeichen")
                return content
            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Transkripts mit alternativer Codierung: {str(e)}")
                raise
                
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
            Sprachcode ('en', 'de', oder andere)
        """
        # Heuristik für Spracherkennung basierend auf typischen Wörtern und Sprachmustern
        language_patterns = {
            'de': ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'auf', 'nicht', 'auch', 'eine', 'wenn', 'aber'],
            'en': ['the', 'and', 'is', 'of', 'to', 'in', 'that', 'it', 'for', 'with', 'as', 'this', 'but'],
            'fr': ['le', 'la', 'les', 'et', 'est', 'en', 'que', 'qui', 'pour', 'dans', 'un', 'une', 'avec']
            # Weitere Sprachen könnten hier hinzugefügt werden
        }
        
        # Zähle Vorkommen typischer Wörter jeder Sprache
        scores = {}
        text_lower = ' ' + text.lower() + ' '  # Leerzeichen hinzufügen für genauere Wortgrenzen
        
        for lang, words in language_patterns.items():
            # Regulärer Ausdruck für genaue Wortübereinstimmungen mit Wortgrenzen
            word_counts = sum(len(re.findall(rf'\b{word}\b', text_lower)) for word in words)
            # Normalisiere nach Anzahl der Wörter in der Sprachmustersammlung
            scores[lang] = word_counts / len(words)
        
        # Ermittle die Sprache mit dem höchsten Score
        detected_lang = max(scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Sprache erkannt: {detected_lang.upper()}")
        return detected_lang
    
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
        # Beispiele: [00:01], [1:23], [01:23:45]
        text = re.sub(r'\[\d+:\d+(?::\d+)?\]', '', text)
        
        # Entferne Sprechermarkierungen
        # Beispiele: [Sprecher 1], [Max Mustermann], [Moderator]
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Entferne übermäßige Leerzeichen und Zeilenumbrüche
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Entferne HTML-Tags (falls vorhanden)
        text = re.sub(r'<[^>]+>', '', text)
        
        self.logger.info("Transkript bereinigt")
        return text.strip()
    
    def split_transcript(self, text: str, max_chunk_size: int = None) -> List[str]:
        """
        Teilt das Transkript in kleinere Abschnitte auf, die vom Modell verarbeitet werden können.
        
        Args:
            text: Das Transkript
            max_chunk_size: Maximale Zeichenanzahl pro Chunk (überschreibt Konfiguration)
            
        Returns:
            Liste von Textabschnitten
        """
        chunk_size = max_chunk_size or self.chunk_size
        self.logger.info(f"Teile Transkript in Abschnitte (max {chunk_size} Zeichen)...")
        
        # Intelligente Aufteilung an Absätzen
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Wenn der Absatz sehr lang ist, teile ihn weiter auf
            if len(paragraph) > chunk_size:
                # Ergänze den aktuellen Chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Teile den langen Absatz in Sätze
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) < chunk_size:
                        temp_chunk += sentence + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "
                
                if temp_chunk:
                    current_chunk = temp_chunk
            
            # Normale Absatzbehandlung
            elif len(current_chunk) + len(paragraph) < chunk_size:
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
    
    def extract_knowledge_from_chunk(self, chunk: str, language: str, task_id: str = None) -> Dict[str, List[str]]:
        """
        Extrahiert strukturiertes Wissen aus einem Transkriptabschnitt.
        
        Args:
            chunk: Ein Abschnitt des Transkripts
            language: Sprachcode ('en', 'de', etc.)
            task_id: Optional, Kennung für Fortschrittsbericht
            
        Returns:
            Wörterbuch mit extrahiertem Wissen nach Kategorien
        """
        self.logger.info(f"Extrahiere Wissen aus Transkriptabschnitt ({len(chunk)} Zeichen)...")
        
        # Modell laden, falls nicht bereits geladen
        if not self._load_model():
            self.logger.error("Modell konnte nicht geladen werden")
            return {category: [] for category in self.categories}
        
        # Prompt je nach Sprache erstellen
        if language == 'de':
            prompt = self._create_german_prompt(chunk)
        else:  # Standardmäßig Englisch
            prompt = self._create_english_prompt(chunk)
        
        try:
            # Fortschritt melden
            if task_id:
                self._report_progress(task_id, 0.3, {"status": "extracting", "chunk_length": len(chunk)})
            
            # Tokenisiere den Prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generiere die Antwort mit kontrollierten Parametern
            output = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            # Fortschritt melden
            if task_id:
                self._report_progress(task_id, 0.7, {"status": "parsing"})
            
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
            
            # Fortschritt melden
            if task_id:
                self._report_progress(task_id, 1.0, {"status": "completed", "items_extracted": sum(len(items) for items in result.values())})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Wissensextraktion: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Fortschritt melden (Fehler)
            if task_id:
                self._report_progress(task_id, 1.0, {"status": "error", "error": str(e)})
            
            return {category: [] for category in self.categories}
    
    def _create_german_prompt(self, chunk: str) -> str:
        """
        Erstellt einen Prompt in deutscher Sprache.
        
        Args:
            chunk: Der zu analysierende Textabschnitt
        
        Returns:
            Prompt für das Modell
        """
        return f"""
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
    
    def _create_english_prompt(self, chunk: str) -> str:
        """
        Erstellt einen Prompt in englischer Sprache.
        
        Args:
            chunk: Der zu analysierende Textabschnitt
        
        Returns:
            Prompt für das Modell
        """
        return f"""
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
    
    def _parse_model_response(self, response: str, language: str) -> Dict[str, List[str]]:
        """
        Parst die Antwort des Modells in ein strukturiertes Format.
        
        Args:
            response: Die Antwort des Modells
            language: Sprachcode ('en', 'de', etc.)
            
        Returns:
            Strukturiertes Wissen nach Kategorien
        """
        result = {category: [] for category in self.categories}
        
        # Mapping von lokalisierten Kategorienamen zu internen Schlüsseln
        category_mapping = {
            # Deutsch
            'trading-strategien': 'trading_strategies',
            'trading strategien': 'trading_strategies',
            'technische indikatoren': 'technical_indicators',
            'risikomanagement': 'risk_management',
            'marktanalyse': 'market_analysis',
            'trading-psychologie': 'trading_psychology',
            'trading psychologie': 'trading_psychology',
            'entry-exit-regeln': 'entry_exit_rules',
            'entry exit regeln': 'entry_exit_rules',
            'position sizing': 'position_sizing',
            
            # Englisch
            'trading strategies': 'trading_strategies',
            'technical indicators': 'technical_indicators',
            'risk management': 'risk_management',
            'market analysis': 'market_analysis',
            'trading psychology': 'trading_psychology',
            'entry-exit rules': 'entry_exit_rules',
            'entry exit rules': 'entry_exit_rules',
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
                # Verschiedene Header-Formate berücksichtigen
                if (line.lower().startswith(f"{category_name}:") or
                    line.lower().startswith(f"- {category_name}:") or
                    line.lower().startswith(f"**{category_name}**") or
                    line.lower() == category_name.upper() or
                    line.lower() == f"# {category_name}" or
                    line.lower() == f"## {category_name}"):
                    
                    current_category = internal_key
                    is_header = True
                    break
            
            if is_header:
                continue
            
            # Wenn wir in einer Kategorie sind und die Zeile ist kein "keine relevanten Informationen"
            no_info_phrases = [
                "keine relevanten informationen", 
                "no relevant information",
                "none", 
                "keine"
            ]
            
            if current_category and not any(phrase in line.lower() for phrase in no_info_phrases):
                # Listenpunkt bereinigen
                if line.startswith('- '):
                    line = line[2:].strip()
                elif line.startswith('* '):
                    line = line[2:].strip()
                elif line.startswith('• '):
                    line = line[2:].strip()
                
                # Nummerierte Liste bereinigen
                if re.match(r'^\d+\.\s', line):
                    line = re.sub(r'^\d+\.\s', '', line)
                
                # Entferne Anführungszeichen, falls vorhanden
                line = line.strip('"\'')
                
                # Füge den Punkt zur entsprechenden Kategorie hinzu, wenn er nicht leer ist
                if line and len(line) > 5:  # Minimale Längenbeschränkung
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
        file_path = Path(file_path)
        task_id = f"transcript_{file_path.stem}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            self.logger.info(f"Starte Verarbeitung der Transkriptdatei: {file_path}")
            self._report_progress(task_id, 0.0, {"status": "starting", "file": str(file_path)})
            
            # Prüfen, ob im Cache
            cache_key = str(file_path.absolute())
            if cache_key in self.results_cache:
                self.logger.info(f"Verwende gecachte Ergebnisse für {file_path}")
                self._report_progress(task_id, 1.0, {"status": "completed", "from_cache": True})
                return self.results_cache[cache_key]
            
            # Lade und bereinige das Transkript
            self._report_progress(task_id, 0.1, {"status": "loading"})
            transcript = self.load_transcript(file_path)
            cleaned_transcript = self.clean_transcript(transcript)
            
            # Erkenne die Sprache
            language = self.detect_language(cleaned_transcript)
            self._report_progress(task_id, 0.2, {"status": "processing", "language": language})
            
            # Teile das Transkript in Abschnitte
            chunks = self.split_transcript(cleaned_transcript)
            self._report_progress(task_id, 0.3, {"status": "splitting", "chunks": len(chunks)})
            
            # Extrahiere Wissen aus jedem Abschnitt
            all_knowledge = {category: [] for category in self.categories}
            
            # Entscheide, ob parallel oder sequentiell verarbeitet wird
            if self.parallel_processing and len(chunks) > 1:
                self._process_chunks_parallel(chunks, language, all_knowledge, task_id)
            else:
                self._process_chunks_sequential(chunks, language, all_knowledge, task_id)
            
            # Entferne Duplikate und sortiere
            self._report_progress(task_id, 0.9, {"status": "finalizing"})
            for category in self.categories:
                # Entferne exakte Duplikate
                unique_items = list(set(all_knowledge[category]))
                
                # Sortiere nach Länge (kürzere Einträge zuerst)
                all_knowledge[category] = sorted(unique_items, key=len)
            
            # Speichere das extrahierte Wissen
            output_file = self.knowledge_path / f"{file_path.stem}_knowledge.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_knowledge, f, ensure_ascii=False, indent=4)
            
            # Erstelle Ergebniszusammenfassung
            result = {
                'status': 'success',
                'file': str(file_path),
                'language': language,
                'chunks': len(chunks),
                'output_file': str(output_file),
                'knowledge_items': {category: len(items) for category, items in all_knowledge.items()},
                'total_items': sum(len(items) for items in all_knowledge.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            # In Cache speichern
            self.results_cache[cache_key] = result
            
            self._report_progress(task_id, 1.0, {"status": "completed", "result": result})
            
            self.logger.info(f"Transkript erfolgreich verarbeitet: {result['total_items']} Wissenseinträge extrahiert")
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Transkript-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            error_result = {
                'status': 'error',
                'file': str(file_path) if 'file_path' in locals() else None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self._report_progress(task_id, 1.0, {"status": "error", "error": str(e)})
            return error_result
    
    def _process_chunks_sequential(self, chunks: List[str], language: str, all_knowledge: Dict[str, List[str]], task_id: str):
        """
        Verarbeitet Chunks sequentiell.
        
        Args:
            chunks: Liste der Textabschnitte
            language: Sprachcode
            all_knowledge: Dict für das gesammelte Wissen
            task_id: Kennung für Fortschrittsmeldungen
        """
        for i, chunk in enumerate(chunks):
            chunk_task_id = f"{task_id}_chunk_{i}"
            self.logger.info(f"Verarbeite Chunk {i+1}/{len(chunks)}...")
            
            # Berechne den Gesamtfortschritt (0.3 bis 0.9 ist für die Chunk-Verarbeitung reserviert)
            progress = 0.3 + (i / len(chunks)) * 0.6
            self._report_progress(task_id, progress, {"status": "processing_chunk", "chunk": i+1, "total_chunks": len(chunks)})
            
            # Extrahiere Wissen aus dem Chunk
            chunk_knowledge = self.extract_knowledge_from_chunk(chunk, language, chunk_task_id)
            
            # Kombiniere die Ergebnisse
            for category in self.categories:
                all_knowledge[category].extend(chunk_knowledge[category])
    
    def _process_chunks_parallel(self, chunks: List[str], language: str, all_knowledge: Dict[str, List[str]], task_id: str):
        """
        Verarbeitet Chunks parallel mit ThreadPoolExecutor.
        
        Args:
            chunks: Liste der Textabschnitte
            language: Sprachcode
            all_knowledge: Dict für das gesammelte Wissen
            task_id: Kennung für Fortschrittsmeldungen
        """
        self.logger.info(f"Starte parallele Verarbeitung von {len(chunks)} Chunks mit {self.max_workers} Workern")
        
        # Thread-Pool für parallele Verarbeitung
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Future-Objekte mit ihren Indizes
            future_to_index = {}
            
            for i, chunk in enumerate(chunks):
                chunk_task_id = f"{task_id}_chunk_{i}"
                future = executor.submit(self.extract_knowledge_from_chunk, chunk, language, chunk_task_id)
                future_to_index[future] = i
            
            # Sammle Ergebnisse, wenn sie verfügbar werden
            completed = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    chunk_knowledge = future.result()
                    
                    # Kombiniere die Ergebnisse
                    for category in self.categories:
                        all_knowledge[category].extend(chunk_knowledge[category])
                    
                    completed += 1
                    progress = 0.3 + (completed / len(chunks)) * 0.6
                    self._report_progress(task_id, progress, {
                        "status": "processing_chunks", 
                        "completed": completed, 
                        "total_chunks": len(chunks)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Fehler bei der Verarbeitung von Chunk {i}: {str(e)}")
    
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
        
        task_id = f"directory_{directory.name}_{datetime.now().strftime('%H%M%S')}"
        self._report_progress(task_id, 0.0, {"status": "starting", "directory": str(directory)})
        
        results = []
        
        try:
            # Finde alle .txt-Dateien im Verzeichnis
            transcript_files = list(directory.glob("*.txt"))
            self.logger.info(f"{len(transcript_files)} Transkript-Dateien gefunden")
            
            self._report_progress(task_id, 0.1, {
                "status": "found_files", 
                "file_count": len(transcript_files)
            })
            
            # Entscheide, ob parallel oder sequentiell verarbeitet wird
            if self.parallel_processing and len(transcript_files) > 1:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(transcript_files))) as executor:
                    future_to_file = {executor.submit(self.process_transcript, file_path): file_path for file_path in transcript_files}
                    
                    completed = 0
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            completed += 1
                            progress = 0.1 + (completed / len(transcript_files)) * 0.8
                            self._report_progress(task_id, progress, {
                                "status": "processing_files", 
                                "completed": completed, 
                                "total_files": len(transcript_files),
                                "current_file": file_path.name
                            })
                            
                        except Exception as e:
                            self.logger.error(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
                            results.append({
                                'status': 'error',
                                'file': str(file_path),
                                'error': str(e)
                            })
            else:
                # Sequentielle Verarbeitung
                for i, file_path in enumerate(transcript_files):
                    self.logger.info(f"Verarbeite Datei {i+1}/{len(transcript_files)}: {file_path}")
                    
                    progress = 0.1 + (i / len(transcript_files)) * 0.8
                    self._report_progress(task_id, progress, {
                        "status": "processing_file", 
                        "file": i+1, 
                        "total_files": len(transcript_files),
                        "current_file": file_path.name
                    })
                    
                    try:
                        result = self.process_transcript(file_path)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
                        results.append({
                            'status': 'error',
                            'file': str(file_path),
                            'error': str(e)
                        })
                    
                    # Kurze Pause zwischen Dateien, um das Modell nicht zu überlasten
                    time.sleep(1)
            
            # Zusammenfassung
            success_count = sum(1 for r in results if r.get('status') == 'success')
            total_items = sum(r.get('total_items', 0) for r in results if r.get('status') == 'success')
            
            self.logger.info(f"Verarbeitung abgeschlossen: {success_count} von {len(results)} erfolgreich")
            self.logger.info(f"Insgesamt {total_items} Wissenseinträge extrahiert")
            
            self._report_progress(task_id, 1.0, {
                "status": "completed", 
                "success_count": success_count,
                "total_files": len(results),
                "total_items": total_items
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verzeichnis-Verarbeitung: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            self._report_progress(task_id, 1.0, {"status": "error", "error": str(e)})
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
        
        task_id = f"combine_{datetime.now().strftime('%H%M%S')}"
        self._report_progress(task_id, 0.0, {"status": "starting"})
        
        try:
            # Finde alle JSON-Dateien im knowledge-Verzeichnis
            knowledge_files = list(self.knowledge_path.glob("*_knowledge.json"))
            self.logger.info(f"{len(knowledge_files)} Wissensdateien gefunden")
            
            self._report_progress(task_id, 0.1, {"status": "found_files", "file_count": len(knowledge_files)})
            
            if not knowledge_files:
                return {'status': 'error', 'message': 'Keine Wissensdateien gefunden'}
            
            # Kombinierte Wissensbasis initialisieren
            combined_knowledge = {category: [] for category in self.categories}
            
            # Alle Dateien durchgehen und Wissen kombinieren
            for i, file_path in enumerate(knowledge_files):
                progress = 0.1 + (i / len(knowledge_files)) * 0.7
                self._report_progress(task_id, progress, {
                    "status": "combining", 
                    "file": i+1, 
                    "total_files": len(knowledge_files),
                    "current_file": file_path.name
                })
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        knowledge = json.load(f)
                        
                        # Kategorien kombinieren
                        for category in self.categories:
                            if category in knowledge and isinstance(knowledge[category], list):
                                combined_knowledge[category].extend(knowledge[category])
                    except json.JSONDecodeError:
                        self.logger.error(f"Fehler beim Parsen der JSON-Datei {file_path}")
            
            self._report_progress(task_id, 0.8, {"status": "deduplicating"})
            
            # Duplikate entfernen und sortieren
            for category in self.categories:
                # Konvertiere in Set für schnellere Duplikatentfernung
                items_set = set(combined_knowledge[category])
                
                # Entferne Duplikate mit ähnlichem Inhalt (case-insensitive)
                items_lower = {}
                unique_items = []
                
                for item in items_set:
                    item_lower = item.lower().strip()
                    if item_lower not in items_lower:
                        items_lower[item_lower] = item
                        unique_items.append(item)
                
                # Nach Länge sortieren (kürzere Einträge zuerst)
                combined_knowledge[category] = sorted(unique_items, key=len)
            
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
            
            self._report_progress(task_id, 1.0, {
                "status": "completed", 
                "total_items": stats['total_items'],
                "output_file": str(output_path)
            })
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Kombinieren der Wissensdateien: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            self._report_progress(task_id, 1.0, {"status": "error", "error": str(e)})
            
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
        
        task_id = f"rules_{datetime.now().strftime('%H%M%S')}"
        self._report_progress(task_id, 0.0, {"status": "starting"})
        
        try:
            # Modell laden, falls nicht bereits geladen
            if not self._load_model():
                error_msg = "Modell konnte nicht geladen werden"
                self.logger.error(error_msg)
                self._report_progress(task_id, 1.0, {"status": "error", "error": error_msg})
                return {'status': 'error', 'message': error_msg}
            
            # Kombinierte Wissensbasis laden
            combined_file = self.knowledge_path / "combined_knowledge.json"
            if not combined_file.exists():
                # Erstelle die kombinierte Wissensbasis, falls sie noch nicht existiert
                self.logger.info("Kombinierte Wissensbasis nicht gefunden, erstelle sie...")
                combine_result = self.combine_knowledge_files()
                
                if combine_result['status'] != 'success':
                    error_msg = "Kombinierte Wissensbasis konnte nicht erstellt werden"
                    self.logger.error(error_msg)
                    self._report_progress(task_id, 1.0, {"status": "error", "error": error_msg})
                    return {'status': 'error', 'message': error_msg}
            
            self._report_progress(task_id, 0.2, {"status": "loading_knowledge"})
            
            with open(combined_file, 'r', encoding='utf-8') as f:
                knowledge = json.load(f)
            
            # Prompt erstellen
            knowledge_text = ""
            for category in self.categories:
                if category in knowledge and knowledge[category]:
                    category_name = category.replace('_', ' ').title()
                    knowledge_text += f"\n{category_name}:\n"
                    
                    # Limitiere auf 20 Einträge pro Kategorie, um Prompt-Länge zu begrenzen
                    for item in knowledge[category][:20]:
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
            
            self._report_progress(task_id, 0.3, {"status": "generating_rules"})
            
            # Generiere Trading-Regeln mit dem Modell
            self.logger.info("Generiere Trading-Regeln mit dem Modell...")
            
            # Tokenisiere den Prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generiere die Antwort
            output = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.1,  # Niedrigere Temperatur für präzisere Regeln
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            self._report_progress(task_id, 0.7, {"status": "parsing_rules"})
            
            # Dekodiere die Antwort
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
            
            total_rules = sum(len(rules_list) for rules_list in rules.values())
            
            self._report_progress(task_id, 1.0, {
                "status": "completed", 
                "total_rules": total_rules,
                "rules_per_category": {k: len(v) for k, v in rules.items()}
            })
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'rules': rules,
                'total_rules': total_rules
            }
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Generierung von Trading-Regeln: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            self._report_progress(task_id, 1.0, {"status": "error", "error": str(e)})
            
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
                # Verschiedene Header-Formate berücksichtigen
                if (re.search(rf'^\d*\.?\s*{header}', line.lower()) or
                    line.lower().startswith(f"# {header}") or
                    line.lower().startswith(f"## {header}") or
                    line.lower().startswith(f"**{header}**") or
                    line.lower() == header.upper()):
                    
                    current_category = key
                    is_header = True
                    break
            
            if is_header:
                continue
            
            # Wenn wir in einer Kategorie sind und die Zeile ein Regelpunkt ist
            if current_category and (line.startswith('-') or line.startswith('*') or 
                                     re.match(r'^\d+\.', line) or line.startswith('•')):
                
                # Listenpunkt bereinigen
                if line.startswith('-'):
                    line = line[1:].strip()
                elif line.startswith('*'):
                    line = line[1:].strip()
                elif line.startswith('•'):
                    line = line[1:].strip()
                else:
                    # Entferne Nummerierung (z.B. "1. ")
                    line = re.sub(r'^\d+\.\s*', '', line)
                
                # Entferne verschiedene Rahmenbegrenzungen
                line = line.strip('"`\'')
                
                # Nur nicht-leere Regeln hinzufügen
                if line:
                    categories[current_category].append(line)
        
        return categories

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'model_path': "google/gemma-3-8b-instruction",
        'device': "auto",
        'data_path': "data",
        'parallel_processing': True,
        'max_workers': 2
    }
    
    # TranscriptProcessor initialisieren
    processor = TranscriptProcessor(config)
    
    # Beispiel für einen Fortschritts-Callback
    def progress_callback(task_id, progress, info):
        print(f"Task {task_id}: {progress*100:.1f}% - {info.get('status', '')}")
    
    processor.register_progress_callback(progress_callback)
    
    # Beispiel für die Verarbeitung eines Transkripts
    # result = processor.process_transcript("data/transcripts/example_transcript.txt")
    # print(f"Ergebnis: {result}")
    
    # Beispiel für die Verarbeitung aller Transkripte im Verzeichnis
    # results = processor.process_directory()
    # print(f"{len(results)} Transkripte verarbeitet")
    
    # Beispiel für die Kombination aller Wissensdateien
    # combined = processor.combine_knowledge_files()
    # print(f"Kombinierte Wissensbasis: {combined}")
    
    # Beispiel für die Generierung von Trading-Regeln
    # rules = processor.generate_trading_rules()
    # print(f"Trading-Regeln: {rules}")

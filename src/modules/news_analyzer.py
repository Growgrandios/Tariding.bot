# news_analyzer.py

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from transformers import pipeline
import re
import time
from concurrent.futures import ThreadPoolExecutor

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/news_analyzer.log"),
        logging.StreamHandler()
    ]
)

class NewsAnalyzer:
    """
    Sammelt und analysiert Krypto-News aus verschiedenen Quellen 
    und bewertet deren Marktauswirkungen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den NewsAnalyzer.
        
        Args:
            config: Konfigurationseinstellungen
        """
        self.logger = logging.getLogger("NewsAnalyzer")
        self.logger.info("Initialisiere NewsAnalyzer...")
        
        # Konfiguration laden
        self.config = config or {}
        
        # API-Schlüssel
        self.api_keys = {
            'newsapi': self.config.get('newsapi_key', os.getenv('NEWS_API_KEY', '')),
            'cryptopanic': self.config.get('cryptopanic_key', os.getenv('CRYPTOPANIC_API_KEY', '')),
            'cryptocompare': self.config.get('cryptocompare_key', os.getenv('CRYPTOCOMPARE_API_KEY', ''))
        }
        
        # News-Quellen
        self.news_sources = {
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/',
            'cryptocompare': 'https://min-api.cryptocompare.com/data/v2/news/',
            'newsapi': 'https://newsapi.org/v2/everything'
        }
        
        # Sentiment-Analyse-Modell laden
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="finiteautomata/bertweet-base-sentiment-analysis"
            )
            self.logger.info("Sentiment-Analyse-Modell erfolgreich geladen")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Sentiment-Modells: {str(e)}")
            self.sentiment_analyzer = None
        
        # Cache für News
        self.news_cache = {
            'last_update': datetime.now() - timedelta(hours=24),
            'articles': []
        }
        
        # Keywords für wichtige Events
        self.important_keywords = {
            'regulatory': [
                'regulation', 'regulatory', 'ban', 'law', 'illegal', 'sec', 'cftc', 
                'compliance', 'license', 'framework', 'legalization'
            ],
            'market_moving': [
                'crash', 'surge', 'skyrocket', 'plummet', 'ATH', 'all-time high', 
                'all-time low', 'correction', 'bubble', 'bear market', 'bull market'
            ],
            'adoption': [
                'adoption', 'institutional', 'etf', 'fund', 'custody', 'integration',
                'partnership', 'enterprise', 'merchant', 'payment'
            ],
            'technology': [
                'upgrade', 'fork', 'update', 'vulnerability', 'hack', 'security',
                'protocol', 'scaling', 'layer 2', 'sidechain', 'rollup'
            ]
        }
        
        # Asset-spezifische Keywords
        self.asset_keywords = {
            'BTC': ['bitcoin', 'btc', 'satoshi', 'lightning network'],
            'ETH': ['ethereum', 'eth', 'vitalik', 'buterin', 'dapps', 'smart contract'],
            'BNB': ['binance', 'bnb', 'binance smart chain', 'bsc', 'cz'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada', 'hoskinson'],
            'AVAX': ['avalanche', 'avax'],
            'DOGE': ['dogecoin', 'doge', 'shiba']
        }
        
        self.logger.info("NewsAnalyzer erfolgreich initialisiert")
    
    def refresh_news(self, force: bool = False) -> bool:
        """
        Aktualisiert den News-Cache aus allen konfigurierten Quellen.
        
        Args:
            force: Erzwingt die Aktualisierung unabhängig vom letzten Update
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        # Prüfen, ob eine Aktualisierung notwendig ist
        now = datetime.now()
        if not force and (now - self.news_cache['last_update']).seconds < 3600:  # Stündliche Updates
            self.logger.debug("News-Cache ist noch aktuell, keine Aktualisierung notwendig")
            return True
        
        self.logger.info("Aktualisiere News-Cache...")
        
        all_articles = []
        
        # Parallel von allen Quellen abrufen
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(self._fetch_from_source, source): source
                for source in self.news_sources.keys()
            }
            
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    articles = future.result()
                    self.logger.info(f"Von {source} wurden {len(articles)} Artikel abgerufen")
                    all_articles.extend(articles)
                except Exception as e:
                    self.logger.error(f"Fehler beim Abrufen von {source}: {str(e)}")
        
        # Duplikate entfernen
        unique_articles = self._deduplicate_articles(all_articles)
        self.logger.info(f"Nach Duplikatentfernung verbleiben {len(unique_articles)} Artikel")
        
        # Sentiment-Analyse durchführen
        if self.sentiment_analyzer:
            try:
                analyzed_articles = self._analyze_sentiment(unique_articles)
                self.news_cache['articles'] = analyzed_articles
            except Exception as e:
                self.logger.error(f"Fehler bei der Sentiment-Analyse: {str(e)}")
                self.news_cache['articles'] = unique_articles
        else:
            self.news_cache['articles'] = unique_articles
        
        # Cache-Zeitstempel aktualisieren
        self.news_cache['last_update'] = now
        
        return True
    
    def _fetch_from_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Ruft News von einer spezifischen Quelle ab.
        
        Args:
            source: Name der Quelle ('cryptopanic', 'cryptocompare', etc.)
            
        Returns:
            Liste von Artikeln
        """
        articles = []
        
        if source == 'cryptopanic':
            url = f"{self.news_sources['cryptopanic']}?auth_token={self.api_keys['cryptopanic']}&public=true"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for result in data.get('results', []):
                    articles.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'source': result.get('source', {}).get('title', 'CryptoPanic'),
                        'published_at': result.get('published_at', ''),
                        'currencies': [c.get('code', '') for c in result.get('currencies', [])],
                        'content': result.get('title', '')  # CryptoPanic provides only titles
                    })
        
        elif source == 'cryptocompare':
            url = f"{self.news_sources['cryptocompare']}?api_key={self.api_keys['cryptocompare']}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('Data', []):
                    articles.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'source': item.get('source', 'CryptoCompare'),
                        'published_at': datetime.fromtimestamp(item.get('published_on', 0)).isoformat(),
                        'currencies': item.get('categories', '').split('|'),
                        'content': item.get('body', '')
                    })
        
        elif source == 'newsapi':
            # Query für Krypto-News
            url = f"{self.news_sources['newsapi']}?q=cryptocurrency OR bitcoin OR crypto&apiKey={self.api_keys['newsapi']}&language=en&sortBy=publishedAt"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    # Extrahiere erwähnte Kryptowährungen
                    currencies = []
                    content = article.get('content', '') or article.get('description', '')
                    for asset, keywords in self.asset_keywords.items():
                        for keyword in keywords:
                            if keyword.lower() in content.lower() or keyword.lower() in article.get('title', '').lower():
                                currencies.append(asset)
                                break
                    
                    articles.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'News API'),
                        'published_at': article.get('publishedAt', ''),
                        'currencies': list(set(currencies)),  # Duplikate entfernen
                        'content': content
                    })
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Entfernt Duplikate aus der Artikelliste basierend auf Titel-Ähnlichkeit.
        
        Args:
            articles: Liste von Artikeln
            
        Returns:
            Liste von eindeutigen Artikeln
        """
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            
            # Grundlegende Normalisierung
            normalized_title = re.sub(r'[^\w\s]', '', title)
            normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()
            
            # Kurze Titel überspringen
            if len(normalized_title) < 10:
                continue
            
            # Prüfen, ob wir einen ähnlichen Titel bereits gesehen haben
            is_duplicate = False
            for seen_title in seen_titles:
                # Einfache Ähnlichkeitsmetrik basierend auf gemeinsamen Wörtern
                if self._jaccard_similarity(normalized_title, seen_title) > 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Berechnet die Jaccard-Ähnlichkeit zwischen zwei Strings.
        
        Args:
            str1: Erster String
            str2: Zweiter String
            
        Returns:
            Ähnlichkeitswert zwischen 0 und 1
        """
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / max(1, union)
    
    def _analyze_sentiment(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Führt Sentiment-Analyse für jeden Artikel durch.
        
        Args:
            articles: Liste von Artikeln
            
        Returns:
            Artikel mit hinzugefügten Sentiment-Scores
        """
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('content', '')
            
            # Begrenzen der Textlänge für die Analyse
            text = text[:512]
            
            if text and self.sentiment_analyzer:
                try:
                    result = self.sentiment_analyzer(text)
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    # Mapping auf -1 bis 1 Skala
                    sentiment_score = 0
                    if label == 'POS':
                        sentiment_score = score
                    elif label == 'NEG':
                        sentiment_score = -score
                    
                    article['sentiment'] = {
                        'score': sentiment_score,
                        'label': label,
                        'magnitude': abs(sentiment_score)
                    }
                except Exception as e:
                    self.logger.warning(f"Fehler bei der Sentiment-Analyse eines Artikels: {str(e)}")
                    article['sentiment'] = {'score': 0, 'label': 'NEUTRAL', 'magnitude': 0}
            else:
                article['sentiment'] = {'score': 0, 'label': 'NEUTRAL', 'magnitude': 0}
        
        return articles
    
    def get_recent_news(self, hours: int = 24, assets: List[str] = None, min_sentiment: float = None) -> List[Dict[str, Any]]:
        """
        Gibt aktuelle News zurück, optional gefiltert nach Assets und Sentiment.
        
        Args:
            hours: Zeitraum in Stunden
            assets: Liste von Asset-Symbolen (z.B. ['BTC', 'ETH'])
            min_sentiment: Minimaler absoluter Sentiment-Wert (für wichtige News)
            
        Returns:
            Liste von gefilterten News-Artikeln
        """
        # News-Cache aktualisieren
        self.refresh_news()
        
        now = datetime.now()
        time_threshold = now - timedelta(hours=hours)
        
        filtered_news = []
        
        for article in self.news_cache['articles']:
            # Zeitfilter
            published_at = article.get('published_at', '')
            if published_at:
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    if pub_date < time_threshold:
                        continue
                except (ValueError, TypeError):
                    # Bei Datumsparsing-Fehlern überspringen wir nicht
                    pass
            
            # Asset-Filter
            if assets:
                currencies = article.get('currencies', [])
                if not any(asset in currencies for asset in assets):
                    # Wenn keine direkte Übereinstimmung, prüfe im Titel und Inhalt
                    title_content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
                    if not any(any(kw in title_content for kw in self.asset_keywords.get(asset, [])) for asset in assets):
                        continue
            
            # Sentiment-Filter
            if min_sentiment is not None:
                sentiment = article.get('sentiment', {}).get('magnitude', 0)
                if sentiment < min_sentiment:
                    continue
            
            filtered_news.append(article)
        
        # Nach Datum sortieren, neueste zuerst
        filtered_news.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        return filtered_news
    
    def get_market_sentiment(self, asset: str = None) -> Dict[str, Any]:
        """
        Berechnet die Gesamtmarkt-Stimmung oder für ein bestimmtes Asset.
        
        Args:
            asset: Optionales Asset-Symbol (z.B. 'BTC')
            
        Returns:
            Dictionary mit Sentiment-Informationen
        """
        # News-Cache aktualisieren
        self.refresh_news()
        
        # News der letzten 24 Stunden
        recent_news = self.get_recent_news(hours=24, assets=[asset] if asset else None)
        
        if not recent_news:
            return {
                'asset': asset or 'MARKET',
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'news_count': 0,
                'confidence': 0
            }
        
        # Sentiment-Werte sammeln
        sentiment_scores = [
            article.get('sentiment', {}).get('score', 0) 
            for article in recent_news
            if 'sentiment' in article
        ]
        
        if not sentiment_scores:
            return {
                'asset': asset or 'MARKET',
                'sentiment_score': 0,
                'sentiment_label': 'NEUTRAL',
                'news_count': len(recent_news),
                'confidence': 0
            }
        
        # Durchschnittlichen Sentiment-Score berechnen
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Label basierend auf dem Score
        label = 'NEUTRAL'
        if avg_sentiment > 0.15:
            label = 'BULLISH'
        elif avg_sentiment > 0.05:
            label = 'SLIGHTLY_BULLISH'
        elif avg_sentiment < -0.15:
            label = 'BEARISH'
        elif avg_sentiment < -0.05:
            label = 'SLIGHTLY_BEARISH'
        
        # Konfidenz basierend auf Anzahl der News
        confidence = min(1.0, len(recent_news) / 20)  # Max-Konfidenz bei 20+ News
        
        return {
            'asset': asset or 'MARKET',
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'news_count': len(recent_news),
            'confidence': confidence
        }
    
    def detect_market_events(self) -> List[Dict[str, Any]]:
        """
        Erkennt wichtige Marktereignisse basierend auf aktuellen News.
        
        Returns:
            Liste von erkannten Ereignissen
        """
        # News-Cache aktualisieren
        self.refresh_news()
        
        # Nur stark positive oder negative News der letzten 12 Stunden
        significant_news = self.get_recent_news(hours=12, min_sentiment=0.4)
        
        events = []
        
        # Ereigniskategorien
        event_types = {
            'regulatory': {'keywords': self.important_keywords['regulatory'], 'threshold': 2},
            'market_moving': {'keywords': self.important_keywords['market_moving'], 'threshold': 2},
            'adoption': {'keywords': self.important_keywords['adoption'], 'threshold': 3},
            'technology': {'keywords': self.important_keywords['technology'], 'threshold': 2}
        }
        
        # Zählen, wie oft jede Kategorie in den News vorkommt
        category_counts = {category: 0 for category in event_types}
        category_articles = {category: [] for category in event_types}
        
        for article in significant_news:
            text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
            
            for category, config in event_types.items():
                keywords = config['keywords']
                for keyword in keywords:
                    if keyword.lower() in text:
                        category_counts[category] += 1
                        category_articles[category].append(article)
                        break  # Nur einmal pro Artikel zählen
        
        # Ereignisse basierend auf Schwellenwerten erkennen
        for category, config in event_types.items():
            if category_counts[category] >= config['threshold']:
                # Betroffene Assets identifizieren
                affected_assets = {}
                for article in category_articles[category]:
                    currencies = article.get('currencies', [])
                    for currency in currencies:
                        affected_assets[currency] = affected_assets.get(currency, 0) + 1
                
                # Hauptbetroffene Assets (Top 3)
                top_assets = sorted(affected_assets.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Durchschnittliches Sentiment für diese Ereigniskategorie
                sentiment_scores = [
                    article.get('sentiment', {}).get('score', 0) 
                    for article in category_articles[category]
                ]
                avg_sentiment = sum(sentiment_scores) / max(1, len(sentiment_scores))
                
                events.append({
                    'event_type': category,
                    'event_count': category_counts[category],
                    'sentiment_score': avg_sentiment,
                    'sentiment_label': 'BULLISH' if avg_sentiment > 0.1 else 'BEARISH' if avg_sentiment < -0.1 else 'NEUTRAL',
                    'affected_assets': [asset for asset, _ in top_assets],
                    'title': f"{category.capitalize()} event detected",
                    'articles': [{'title': a.get('title', ''), 'url': a.get('url', '')} for a in category_articles[category][:3]]
                })
        
        return events
    
    def get_asset_news_summary(self, asset: str) -> Dict[str, Any]:
        """
        Erstellt eine Zusammenfassung aller News für ein bestimmtes Asset.
        
        Args:
            asset: Asset-Symbol (z.B. 'BTC')
            
        Returns:
            Dictionary mit News-Zusammenfassung
        """
        # News-Cache aktualisieren
        self.refresh_news()
        
        # News für das Asset abrufen
        asset_news = self.get_recent_news(hours=48, assets=[asset])
        
        if not asset_news:
            return {
                'asset': asset,
                'news_count': 0,
                'summary': f"No recent news found for {asset}."
            }
        
        # Sentiment-Analyse
        sentiment = self.get_market_sentiment(asset)
        
        # Wichtige Themen identifizieren
        topic_counts = {
            'regulatory': 0,
            'market_moving': 0,
            'adoption': 0,
            'technology': 0
        }
        
        for article in asset_news:
            text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
            
            for topic, keywords in self.important_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        topic_counts[topic] += 1
                        break  # Nur einmal pro Artikel zählen
        
        # Wichtigste Themen (Top 2)
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Meistgenannte andere Assets
        other_assets = {}
        for article in asset_news:
            currencies = article.get('currencies', [])
            for currency in currencies:
                if currency != asset and currency:
                    other_assets[currency] = other_assets.get(currency, 0) + 1
        
        related_assets = sorted(other_assets.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Die wichtigsten 5 Artikel basierend auf Sentiment-Magnitude
        top_articles = sorted(asset_news, key=lambda x: abs(x.get('sentiment', {}).get('score', 0)), reverse=True)[:5]
        
        return {
            'asset': asset,
            'news_count': len(asset_news),
            'sentiment': sentiment,
            'top_topics': [topic for topic, count in top_topics if count > 0],
            'related_assets': [asset for asset, count in related_assets],
            'important_articles': [
                {
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'sentiment_score': article.get('sentiment', {}).get('score', 0),
                    'published_at': article.get('published_at', '')
                }
                for article in top_articles
            ]
        }

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'newsapi_key': os.getenv('NEWS_API_KEY', ''),
        'cryptopanic_key': os.getenv('CRYPTOPANIC_API_KEY', ''),
        'cryptocompare_key': os.getenv('CRYPTOCOMPARE_API_KEY', '')
    }
    
    # NewsAnalyzer initialisieren
    news_analyzer = NewsAnalyzer(config)
    
    # News abrufen und analysieren
    news_analyzer.refresh_news()
    
    # Markt-Sentiment abrufen
    market_sentiment = news_analyzer.get_market_sentiment()
    print(f"Market Sentiment: {market_sentiment}")
    
    # BTC-News zusammenfassen
    btc_summary = news_analyzer.get_asset_news_summary('BTC')
    print(f"BTC News Summary: {btc_summary}")
    
    # Wichtige Marktereignisse erkennen
    events = news_analyzer.detect_market_events()
    print(f"Detected Market Events: {events}")

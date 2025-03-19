# token_scanner.py

import os
import json
import logging
import time
import pandas as pd
import numpy as np
import requests
from web3 import Web3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import asyncio
import aiohttp
from statistics import mean, median

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/token_scanner.log"),
        logging.StreamHandler()
    ]
)

class TokenScanner:
    """
    Überwacht DEXs auf neue Token-Listings und analysiert Liquidität, 
    Handelsmuster und weitere Metriken.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den TokenScanner.
        
        Args:
            config: Konfigurationseinstellungen
        """
        self.logger = logging.getLogger("TokenScanner")
        self.logger.info("Initialisiere TokenScanner...")
        
        # Konfiguration laden
        self.config = config or {}
        
        # API-Schlüssel
        self.api_keys = {
            'etherscan': self.config.get('etherscan_key', os.getenv('ETHERSCAN_API_KEY', '')),
            'bscscan': self.config.get('bscscan_key', os.getenv('BSCSCAN_API_KEY', '')),
            'dexscreener': self.config.get('dexscreener_key', os.getenv('DEXSCREENER_API_KEY', ''))
        }
        
        # Unterstützte Blockchains und DEXs
        self.blockchains = {
            'ethereum': {
                'name': 'Ethereum',
                'rpc_url': self.config.get('eth_rpc_url', 'https://mainnet.infura.io/v3/your-infura-key'),
                'explorer_api': 'https://api.etherscan.io/api',
                'factory_addresses': {
                    'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                    'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'
                },
                'dexs': ['uniswap', 'sushiswap']
            },
            'bsc': {
                'name': 'Binance Smart Chain',
                'rpc_url': self.config.get('bsc_rpc_url', 'https://bsc-dataseed.binance.org/'),
                'explorer_api': 'https://api.bscscan.com/api',
                'factory_addresses': {
                    'pancakeswap_v2': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
                },
                'dexs': ['pancakeswap']
            },
            'polygon': {
                'name': 'Polygon',
                'rpc_url': self.config.get('polygon_rpc_url', 'https://polygon-rpc.com'),
                'explorer_api': 'https://api.polygonscan.com/api',
                'factory_addresses': {
                    'quickswap': '0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32'
                },
                'dexs': ['quickswap']
            }
        }
        
        # Web3-Verbindungen initialisieren
        self.web3_connections = {}
        for chain_id, chain_data in self.blockchains.items():
            try:
                w3 = Web3(Web3.HTTPProvider(chain_data['rpc_url']))
                if w3.is_connected():
                    self.web3_connections[chain_id] = w3
                    self.logger.info(f"Web3-Verbindung zu {chain_data['name']} hergestellt")
                else:
                    self.logger.error(f"Verbindung zu {chain_data['name']} nicht möglich")
            except Exception as e:
                self.logger.error(f"Fehler beim Verbinden mit {chain_data['name']}: {str(e)}")
        
        # Token-Cache
        self.token_cache = {
            'new_tokens': [],  # Neueste Token
            'monitored_tokens': [],  # Token unter Beobachtung
            'last_update': datetime.now() - timedelta(hours=24)
        }
        
        # Dexscreener API für Live-Daten
        self.dexscreener_api = 'https://api.dexscreener.com/latest/dex'
        
        # Token-Kontraktvorlagen für die Analyse
        with open('data/contract_templates/token_templates.json', 'r') as f:
            self.token_templates = json.load(f)
        
        # Gefährliche Funktionen im Kontrakt
        self.dangerous_functions = [
            'mintable', 'ownable', 'pausable', 'blacklist', 
            'whitelist', 'tax', 'fee', 'burn', 'reflect'
        ]
        
        # Thread für kontinuierliches Scannen
        self.scanning_thread = None
        self.stop_scanning = False
        
        self.logger.info("TokenScanner erfolgreich initialisiert")
    
    def start_scanning(self, interval_seconds: int = 300):
        """
        Startet kontinuierliches Scannen nach neuen Token.
        
        Args:
            interval_seconds: Intervall zwischen Scans in Sekunden
        """
        if self.scanning_thread and self.scanning_thread.is_alive():
            self.logger.warning("Scanner läuft bereits")
            return
        
        self.stop_scanning = False
        self.scanning_thread = threading.Thread(
            target=self._continuous_scanning,
            args=(interval_seconds,),
            daemon=True
        )
        self.scanning_thread.start()
        self.logger.info(f"Token-Scanner gestartet (Intervall: {interval_seconds}s)")
    
    def stop_scanning(self):
        """Stoppt das kontinuierliche Scannen"""
        self.stop_scanning = True
        if self.scanning_thread:
            self.scanning_thread.join(timeout=10)
            self.logger.info("Token-Scanner gestoppt")
    
    def _continuous_scanning(self, interval_seconds: int):
        """
        Thread-Funktion für kontinuierliches Scannen.
        
        Args:
            interval_seconds: Intervall zwischen Scans
        """
        while not self.stop_scanning:
            try:
                self.scan_new_tokens()
                
                # Überwachte Token aktualisieren
                for token in self.token_cache['monitored_tokens']:
                    self.update_token_metrics(token)
                
                # Warten bis zum nächsten Scan
                time_to_sleep = interval_seconds
                for _ in range(time_to_sleep):
                    if self.stop_scanning:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Fehler beim kontinuierlichen Scanning: {str(e)}")
                time.sleep(60)  # Kurze Pause bei Fehlern
    
    async def scan_new_tokens_async(self):
        """
        Scannt asynchron nach neuen Token auf allen unterstützten DEXs.
        
        Returns:
            Liste neuer Token
        """
        self.logger.info("Starte asynchronen Scan nach neuen Token...")
        new_tokens = []
        
        # Aktuelle Zeit
        current_time = datetime.now()
        last_update = self.token_cache['last_update']
        
        # Nur scannen, wenn die letzte Aktualisierung mindestens 5 Minuten zurückliegt
        if (current_time - last_update).seconds < 300:
            self.logger.debug("Letzter Scan war vor weniger als 5 Minuten, überspringe...")
            return self.token_cache['new_tokens']
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Tasks für alle Blockchains und DEXs erstellen
            for chain_id, chain_data in self.blockchains.items():
                for dex in chain_data['dexs']:
                    tasks.append(self._scan_dex_async(session, chain_id, dex))
            
            # Alle Tasks ausführen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Ergebnisse sammeln
            for result in results:
                if isinstance(result, list):
                    new_tokens.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Fehler beim asynchronen Scannen: {str(result)}")
        
        # Token-Cache aktualisieren
        self.token_cache['new_tokens'] = new_tokens
        self.token_cache['last_update'] = current_time
        
        self.logger.info(f"{len(new_tokens)} neue Token gefunden")
        return new_tokens
    
    async def _scan_dex_async(self, session: aiohttp.ClientSession, chain_id: str, dex: str) -> List[Dict[str, Any]]:
        """
        Scannt asynchron einen bestimmten DEX nach neuen Token.
        
        Args:
            session: Aiohttp Session
            chain_id: Blockchain-ID
            dex: Name der DEX
            
        Returns:
            Liste neuer Token
        """
        new_tokens = []
        
        # Die letzten 24 Stunden
        from_time = int((datetime.now() - timedelta(hours=24)).timestamp())
        
        try:
            # DEXScreener API verwenden für neue Pairings
            url = f"{self.dexscreener_api}/pairs/{chain_id.lower()}/{dex.lower()}?from={from_time}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    for pair in pairs:
                        # Prüfen, ob das Token neu ist (weniger als 24h alt)
                        created_at = pair.get('pairCreatedAt')
                        
                        if created_at:
                            created_time = datetime.fromtimestamp(created_at / 1000)
                            hours_since_creation = (datetime.now() - created_time).total_seconds() / 3600
                            
                            if hours_since_creation <= 24:
                                # Basisdaten des Tokens extrahieren
                                token_data = {
                                    'address': pair.get('baseToken', {}).get('address'),
                                    'name': pair.get('baseToken', {}).get('name'),
                                    'symbol': pair.get('baseToken', {}).get('symbol'),
                                    'chain': chain_id,
                                    'dex': dex,
                                    'created_at': created_time.isoformat(),
                                    'pair_address': pair.get('pairAddress'),
                                    'liquidity_usd': pair.get('liquidity', {}).get('usd', 0),
                                    'price_usd': pair.get('priceUsd', 0),
                                    'volume_24h': pair.get('volume', {}).get('h24', 0),
                                    'txns_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0) + 
                                               pair.get('txns', {}).get('h24', {}).get('sells', 0),
                                    'holders': 0,  # Wird später aktualisiert
                                    'risk_score': 0,  # Wird später berechnet
                                    'verified': False  # Wird später überprüft
                                }
                                
                                # Prüfen, ob dieses Token bereits bekannt ist
                                if not any(t['address'] == token_data['address'] for t in self.token_cache['new_tokens']):
                                    new_tokens.append(token_data)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Scannen von {dex} auf {chain_id}: {str(e)}")
        
        return new_tokens
    
    def scan_new_tokens(self) -> List[Dict[str, Any]]:
        """
        Synchrone Wrapper-Funktion für asynchrones Scannen nach neuen Token.
        
        Returns:
            Liste neuer Token
        """
        try:
            # Event Loop erstellen
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Asynchrones Scannen ausführen
            new_tokens = loop.run_until_complete(self.scan_new_tokens_async())
            
            # Event Loop schließen
            loop.close()
            
            return new_tokens
            
        except Exception as e:
            self.logger.error(f"Fehler beim Scannen nach neuen Token: {str(e)}")
            return []
    
    def analyze_token(self, token_address: str, chain_id: str) -> Dict[str, Any]:
        """
        Analysiert ein Token und bewertet dessen Risikopotenzial.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Dictionary mit Token-Analyse
        """
        if chain_id not in self.web3_connections:
            self.logger.error(f"Keine Web3-Verbindung für {chain_id}")
            return {'error': f"Keine Web3-Verbindung für {chain_id}"}
        
        # Cache prüfen
        for token in self.token_cache['new_tokens'] + self.token_cache['monitored_tokens']:
            if token['address'].lower() == token_address.lower() and token['chain'] == chain_id:
                # Token bereits analysiert, zurückgeben
                return token
        
        self.logger.info(f"Analysiere Token {token_address} auf {chain_id}")
        
        try:
            # Token-Metadaten abrufen
            token_data = self._fetch_token_metadata(token_address, chain_id)
            
            if 'error' in token_data:
                return token_data
            
            # Token-Kontrakt analysieren
            risk_analysis = self._analyze_contract_risk(token_address, chain_id)
            
            # Liquiditäts- und Marktdaten abrufen
            market_data = self._fetch_market_data(token_address, chain_id)
            
            # Risiko-Score basierend auf allen Faktoren berechnen
            risk_score = self._calculate_risk_score(token_data, risk_analysis, market_data)
            
            # Ergebnisse zusammenfassen
            result = {
                **token_data,
                **risk_analysis,
                **market_data,
                'risk_score': risk_score,
                'analysis_time': datetime.now().isoformat()
            }
            
            # Token zur Überwachungsliste hinzufügen, wenn es nicht bereits dort ist
            if not any(t['address'].lower() == token_address.lower() for t in self.token_cache['monitored_tokens']):
                self.token_cache['monitored_tokens'].append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Token-Analyse von {token_address}: {str(e)}")
            return {'error': str(e)}
    
    def _fetch_token_metadata(self, token_address: str, chain_id: str) -> Dict[str, Any]:
        """
        Ruft grundlegende Metadaten für ein Token ab.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Dictionary mit Token-Metadaten
        """
        # ERC20 ABI für grundlegende Funktionen
        erc20_abi = [
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"},
            {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "payable": False, "stateMutability": "view", "type": "function"}
        ]
        
        w3 = self.web3_connections.get(chain_id)
        if not w3:
            return {'error': f"Keine Web3-Verbindung für {chain_id}"}
        
        try:
            # Token-Kontrakt initialisieren
            token_contract = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=erc20_abi)
            
            # Grundlegende Daten abrufen
            name = token_contract.functions.name().call()
            symbol = token_contract.functions.symbol().call()
            decimals = token_contract.functions.decimals().call()
            total_supply = token_contract.functions.totalSupply().call() / (10 ** decimals)
            
            # Holen der Creator-Adresse und Erstellungszeit
            blockchain = self.blockchains.get(chain_id, {})
            explorer_api = blockchain.get('explorer_api', '')
            api_key = self.api_keys.get(f"{chain_id}scan", '')
            
            creation_info = {'creator': 'unknown', 'created_at': 'unknown'}
            
            if explorer_api and api_key:
                # API-Aufruf für Kontraktdetails
                url = f"{explorer_api}?module=contract&action=getcontractcreation&contractaddresses={token_address}&apikey={api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == '1' and data.get('result'):
                        creator = data['result'][0].get('contractCreator')
                        creation_info['creator'] = creator
                        
                        # Ersten Block des Kontrakts abrufen
                        url = f"{explorer_api}?module=account&action=txlist&address={token_address}&startblock=0&endblock=99999999&page=1&offset=1&sort=asc&apikey={api_key}"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('status') == '1' and data.get('result'):
                                timestamp = int(data['result'][0].get('timeStamp', 0))
                                creation_info['created_at'] = datetime.fromtimestamp(timestamp).isoformat()
            
            # Kontraktcode verifiziert?
            is_verified = self._check_contract_verified(token_address, chain_id)
            
            return {
                'address': token_address,
                'name': name,
                'symbol': symbol,
                'decimals': decimals,
                'total_supply': total_supply,
                'chain': chain_id,
                'creator': creation_info['creator'],
                'created_at': creation_info['created_at'],
                'verified': is_verified
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Token-Metadaten für {token_address}: {str(e)}")
            return {'error': str(e)}
    
    def _check_contract_verified(self, token_address: str, chain_id: str) -> bool:
        """
        Prüft, ob der Token-Kontrakt verifiziert ist.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            True, wenn der Kontrakt verifiziert ist, sonst False
        """
        blockchain = self.blockchains.get(chain_id, {})
        explorer_api = blockchain.get('explorer_api', '')
        api_key = self.api_keys.get(f"{chain_id}scan", '')
        
        if not explorer_api or not api_key:
            return False
        
        try:
            # API-Aufruf für Kontraktcode
            url = f"{explorer_api}?module=contract&action=getsourcecode&address={token_address}&apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1' and data.get('result'):
                    # Wenn der Sourcecode vorhanden ist und nicht "Unverified", ist der Kontrakt verifiziert
                    source_code = data['result'][0].get('SourceCode', '')
                    return len(source_code) > 0 and source_code.lower() != "unverified"
            
            return False
            
        except Exception as e:
            self.logger.error(f"Fehler beim Prüfen der Kontraktverifizierung für {token_address}: {str(e)}")
            return False
    
    def _analyze_contract_risk(self, token_address: str, chain_id: str) -> Dict[str, Any]:
        """
        Analysiert den Token-Kontrakt auf potenzielle Risiken.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Dictionary mit Risikomerkmalen
        """
        blockchain = self.blockchains.get(chain_id, {})
        explorer_api = blockchain.get('explorer_api', '')
        api_key = self.api_keys.get(f"{chain_id}scan", '')
        
        # Standardergebnis, wenn keine Analyse möglich ist
        default_result = {
            'risky_functions': [],
            'has_mint_function': False,
            'has_blacklist_function': False,
            'has_fee_manipulation': False,
            'contract_similar_to_scam': False,
            'contract_risk_level': 'unknown'
        }
        
        if not explorer_api or not api_key:
            return default_result
        
        try:
            # API-Aufruf für Kontraktcode
            url = f"{explorer_api}?module=contract&action=getsourcecode&address={token_address}&apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return default_result
                
            data = response.json()
            if data.get('status') != '1' or not data.get('result'):
                return default_result
                
            source_code = data['result'][0].get('SourceCode', '')
            
            if not source_code or source_code.lower() == "unverified":
                # Unverified contracts are high risk
                return {
                    **default_result,
                    'contract_risk_level': 'high'
                }
            
            # Gefährliche Funktionen prüfen
            risky_functions = []
            
            if 'function mint' in source_code.lower():
                risky_functions.append('mint')
                default_result['has_mint_function'] = True
            
            if any(term in source_code.lower() for term in ['blacklist', 'whitelist']):
                risky_functions.append('blacklist/whitelist')
                default_result['has_blacklist_function'] = True
            
            # Fee-Manipulation
            if 'fee' in source_code.lower() and any(term in source_code.lower() for term in ['update', 'change', 'set', 'modify']):
                risky_functions.append('fee_manipulation')
                default_result['has_fee_manipulation'] = True
            
            # Ähnlichkeit mit bekannten Scam-Kontrakten prüfen
            contract_similar_to_scam = False
            for template in self.token_templates.get('scam_signatures', []):
                if template in source_code:
                    contract_similar_to_scam = True
                    break
            
            default_result['contract_similar_to_scam'] = contract_similar_to_scam
            default_result['risky_functions'] = risky_functions
            
            # Risikostufe bestimmen
            if contract_similar_to_scam or len(risky_functions) >= 3:
                risk_level = 'high'
            elif len(risky_functions) >= 1:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            default_result['contract_risk_level'] = risk_level
            
            return default_result
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Kontraktanalyse für {token_address}: {str(e)}")
            return default_result
    
    def _fetch_market_data(self, token_address: str, chain_id: str) -> Dict[str, Any]:
        """
        Ruft Marktdaten für ein Token ab.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Dictionary mit Marktdaten
        """
        # Standardergebnis
        default_result = {
            'price_usd': 0,
            'liquidity_usd': 0,
            'volume_24h': 0,
            'price_change_24h': 0,
            'holders': 0,
            'pairs': []
        }
        
        try:
            # DexScreener API für Live-Marktdaten verwenden
            url = f"{self.dexscreener_api}/tokens/{token_address}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return default_result
                
            data = response.json()
            pairs = data.get('pairs', [])
            
            if not pairs:
                return default_result
            
            # Relevante Paare für diese Chain filtern
            chain_pairs = [p for p in pairs if p.get('chainId') == chain_id]
            
            if not chain_pairs:
                return default_result
            
            # Aggregierte Daten berechnen
            liquidity_usd = sum(p.get('liquidity', {}).get('usd', 0) for p in chain_pairs)
            volume_24h = sum(p.get('volume', {}).get('h24', 0) for p in chain_pairs)
            
            # Gewichteten Durchschnittspreis berechnen
            total_volume = sum(p.get('volume', {}).get('h24', 0) or 0 for p in chain_pairs)
            if total_volume > 0:
                weighted_price = sum((p.get('priceUsd', 0) or 0) * (p.get('volume', {}).get('h24', 0) or 0) for p in chain_pairs) / total_volume
            else:
                weighted_price = chain_pairs[0].get('priceUsd', 0) if chain_pairs else 0
            
            # Durchschnittliche Preisänderung
            price_changes = [p.get('priceChange', {}).get('h24', 0) for p in chain_pairs if p.get('priceChange', {}).get('h24') is not None]
            avg_price_change = mean(price_changes) if price_changes else 0
            
            # Pair-Informationen sammeln
            pair_info = [
                {
                    'dex': p.get('dexId', ''),
                    'pair_address': p.get('pairAddress', ''),
                    'base_token': p.get('baseToken', {}).get('symbol', ''),
                    'quote_token': p.get('quoteToken', {}).get('symbol', ''),
                    'price_usd': p.get('priceUsd', 0),
                    'liquidity_usd': p.get('liquidity', {}).get('usd', 0),
                    'volume_24h': p.get('volume', {}).get('h24', 0)
                }
                for p in chain_pairs
            ]
            
            # Anzahl der Tokenhalter (erfordert zusätzliche API-Aufrufe)
            holders = self._fetch_token_holders(token_address, chain_id)
            
            return {
                'price_usd': weighted_price,
                'liquidity_usd': liquidity_usd,
                'volume_24h': volume_24h,
                'price_change_24h': avg_price_change,
                'holders': holders,
                'pairs': pair_info
            }
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Marktdaten für {token_address}: {str(e)}")
            return default_result
    
    def _fetch_token_holders(self, token_address: str, chain_id: str) -> int:
        """
        Ruft die Anzahl der Tokenhalter ab.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Anzahl der Tokenhalter
        """
        # Diese Funktion ist abhängig vom Explorer und kann Limits haben
        blockchain = self.blockchains.get(chain_id, {})
        explorer_api = blockchain.get('explorer_api', '')
        api_key = self.api_keys.get(f"{chain_id}scan", '')
        
        if not explorer_api or not api_key:
            return 0
        
        try:
            # API-Aufruf für Tokenhalter (nicht alle Explorer unterstützen dies)
            url = f"{explorer_api}?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=1&apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1' and data.get('result'):
                    return int(data.get('result', {}).get('total_holder_count', 0))
            
            # Alternativ bei einigen Explorern
            url = f"{explorer_api}?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Manche Explorer geben die Anzahl der Halter direkt zurück
                data = response.json()
                if data.get('status') == '1' and data.get('result') and isinstance(data.get('result'), dict):
                    return int(data.get('result', {}).get('holder_count', 0))
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Tokenhalter für {token_address}: {str(e)}")
            return 0
    
    def _calculate_risk_score(self, token_data: Dict[str, Any], risk_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Berechnet einen numerischen Risiko-Score für ein Token (0-100, höher = riskanter).
        
        Args:
            token_data: Token-Metadaten
            risk_analysis: Ergebnisse der Kontraktanalyse
            market_data: Marktdaten
            
        Returns:
            Risiko-Score (0-100)
        """
        score = 0
        
        # 1. Kontraktverifizierung (0-20 Punkte)
        if not token_data.get('verified', False):
            score += 20
        
        # 2. Kontraktrisiko basierend auf gefährlichen Funktionen (0-30 Punkte)
        contract_risk_level = risk_analysis.get('contract_risk_level', 'unknown')
        if contract_risk_level == 'high':
            score += 30
        elif contract_risk_level == 'medium':
            score += 15
        elif contract_risk_level == 'low':
            score += 5
        else:  # unknown
            score += 20
        
        # 3. Marktliquidität (0-20 Punkte, weniger Liquidität = mehr Risiko)
        liquidity = market_data.get('liquidity_usd', 0)
        if liquidity < 1000:
            score += 20
        elif liquidity < 10000:
            score += 15
        elif liquidity < 50000:
            score += 10
        elif liquidity < 100000:
            score += 5
        
        # 4. Alter des Tokens (0-10 Punkte, neuer = riskanter)
        created_at = token_data.get('created_at')
        if created_at and created_at != 'unknown':
            try:
                creation_time = datetime.fromisoformat(created_at)
                days_since_creation = (datetime.now() - creation_time).total_seconds() / (24 * 3600)
                
                if days_since_creation < 1:
                    score += 10
                elif days_since_creation < 7:
                    score += 7
                elif days_since_creation < 30:
                    score += 5
                elif days_since_creation < 90:
                    score += 2
            except:
                score += 5  # Bei Fehlern beim Datumsparsing
        else:
            score += 10  # Wenn kein Erstellungsdatum bekannt ist
        
        # 5. Marktaktivität (0-10 Punkte, weniger Aktivität = mehr Risiko)
        volume = market_data.get('volume_24h', 0)
        holders = market_data.get('holders', 0)
        
        if volume < 1000 or holders < 10:
            score += 10
        elif volume < 10000 or holders < 50:
            score += 7
        elif volume < 50000 or holders < 100:
            score += 5
        elif volume < 100000 or holders < 200:
            score += 2
        
        # 6. Kontraktähnlichkeit mit bekannten Scams (0-10 Punkte)
        if risk_analysis.get('contract_similar_to_scam', False):
            score += 10
        
        # Risiko-Score begrenzen
        return min(100, max(0, score))
    
    def update_token_metrics(self, token: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aktualisiert die Metriken für ein überwachtes Token.
        
        Args:
            token: Token-Daten
            
        Returns:
            Aktualisierte Token-Daten
        """
        if not token or 'address' not in token or 'chain' not in token:
            return token
        
        try:
            # Marktdaten aktualisieren
            market_data = self._fetch_market_data(token['address'], token['chain'])
            
            # Token-Daten aktualisieren
            updated_token = {**token, **market_data}
            
            # Risiko-Score neu berechnen
            risk_analysis = updated_token.get('risky_functions', [])
            token_data = {
                'address': updated_token.get('address'),
                'name': updated_token.get('name'),
                'symbol': updated_token.get('symbol'),
                'created_at': updated_token.get('created_at'),
                'verified': updated_token.get('verified', False)
            }
            
            risk_score = self._calculate_risk_score(token_data, updated_token, market_data)
            updated_token['risk_score'] = risk_score
            
            # Aktualisierungszeit
            updated_token['last_updated'] = datetime.now().isoformat()
            
            # Token im Cache aktualisieren
            for i, t in enumerate(self.token_cache['monitored_tokens']):
                if t['address'] == token['address'] and t['chain'] == token['chain']:
                    self.token_cache['monitored_tokens'][i] = updated_token
                    break
            
            return updated_token
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Aktualisierung der Token-Metriken für {token.get('address')}: {str(e)}")
            return token
    
    def get_new_tokens(self, max_age_hours: int = 24, min_liquidity: float = 0, max_risk: float = 100) -> List[Dict[str, Any]]:
        """
        Gibt gefilterte neue Token zurück.
        
        Args:
            max_age_hours: Maximales Alter in Stunden
            min_liquidity: Minimale Liquidität in USD
            max_risk: Maximaler Risiko-Score (0-100)
            
        Returns:
            Liste gefilterter Token
        """
        # Scan starten, falls nötig
        if (datetime.now() - self.token_cache['last_update']).seconds > 3600:
            self.scan_new_tokens()
        
        # Token filtern
        filtered_tokens = []
        current_time = datetime.now()
        
        for token in self.token_cache['new_tokens']:
            # Altersfilter
            created_at = token.get('created_at')
            age_ok = True
            
            if created_at and created_at != 'unknown':
                try:
                    creation_time = datetime.fromisoformat(created_at)
                    hours_since_creation = (current_time - creation_time).total_seconds() / 3600
                    age_ok = hours_since_creation <= max_age_hours
                except:
                    pass  # Bei Fehlern beim Datumsparsing überspringen wir den Filter
            
            # Liquiditätsfilter
            liquidity_ok = token.get('liquidity_usd', 0) >= min_liquidity
            
            # Risikofilter
            risk_ok = token.get('risk_score', 100) <= max_risk
            
            if age_ok and liquidity_ok and risk_ok:
                filtered_tokens.append(token)
        
        # Nach Liquidität sortieren (höchste zuerst)
        filtered_tokens.sort(key=lambda x: x.get('liquidity_usd', 0), reverse=True)
        
        return filtered_tokens
    
    def get_potential_gems(self, max_tokens: int = 10) -> List[Dict[str, Any]]:
        """
        Identifiziert potenzielle "Gem"-Token mit gutem Risiko-Rendite-Profil.
        
        Args:
            max_tokens: Maximale Anzahl zurückzugebender Token
            
        Returns:
            Liste potenzieller Gem-Token
        """
        # Alle Token aus dem Cache holen
        all_tokens = self.token_cache['new_tokens'] + self.token_cache['monitored_tokens']
        
        # Duplikate entfernen
        unique_tokens = []
        seen_addresses = set()
        
        for token in all_tokens:
            addr_chain = f"{token.get('address', '')}-{token.get('chain', '')}"
            if addr_chain not in seen_addresses and 'address' in token:
                seen_addresses.add(addr_chain)
                unique_tokens.append(token)
        
        # Bewertungsscore für jedes Token berechnen
        scored_tokens = []
        
        for token in unique_tokens:
            # Grundlegende Daten
            liquidity = token.get('liquidity_usd', 0)
            volume = token.get('volume_24h', 0)
            holders = token.get('holders', 0)
            risk_score = token.get('risk_score', 50)
            verified = token.get('verified', False)
            
            # Score-Berechnung: höhere Werte sind besser
            if liquidity <= 0 or volume <= 0:
                continue  # Token ohne Liquidität oder Volumen überspringen
            
            # Liquiditäts-/Volumen-Verhältnis (höher = besser, max. 1)
            liq_vol_ratio = min(1.0, volume / max(1, liquidity)) if liquidity > 0 else 0
            
            # Risiko-Faktor (niedriger Risiko-Score = besser)
            risk_factor = 1 - (risk_score / 100)
            
            # Verifizierungsbonus
            verification_bonus = 0.2 if verified else 0
            
            # Halter-Bonus (mehr Halter = besser, max. 0.15)
            holder_bonus = min(0.15, holders / 1000) if holders > 0 else 0
            
            # Gesamtbewertung
            gem_score = (0.4 * liq_vol_ratio) + (0.3 * risk_factor) + verification_bonus + holder_bonus
            
            scored_tokens.append({
                **token,
                'gem_score': gem_score
            })
        
        # Nach Gem-Score sortieren (höchster zuerst)
        scored_tokens.sort(key=lambda x: x.get('gem_score', 0), reverse=True)
        
        return scored_tokens[:max_tokens]
    
    def generate_token_report(self, token_address: str, chain_id: str) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Bericht für ein Token.
        
        Args:
            token_address: Token-Kontraktadresse
            chain_id: Blockchain-ID
            
        Returns:
            Dictionary mit Berichtsdaten
        """
        # Token analysieren oder aus Cache abrufen
        token = self.analyze_token(token_address, chain_id)
        
        if 'error' in token:
            return {'error': token['error']}
        
        # Zusätzliche Daten für den Bericht sammeln
        pairs = token.get('pairs', [])
        price_usd = token.get('price_usd', 0)
        liquidity_usd = token.get('liquidity_usd', 0)
        holders = token.get('holders', 0)
        risk_score = token.get('risk_score', 0)
        
        # Risikoklassifizierung
        risk_category = 'High Risk'
        if risk_score < 30:
            risk_category = 'Low Risk'
        elif risk_score < 60:
            risk_category = 'Medium Risk'
        
        # Übersichtstext generieren
        if risk_score < 30:
            overview = f"{token.get('name', '')} ({token.get('symbol', '')}) appears to be a lower risk token with good liquidity and trading activity."
        elif risk_score < 60:
            overview = f"{token.get('name', '')} ({token.get('symbol', '')}) shows moderate risk factors. Conduct further research before investing."
        else:
            overview = f"{token.get('name', '')} ({token.get('symbol', '')}) displays multiple high-risk indicators. Extreme caution is advised."
        
        # Berichtzusammenfassung
        report = {
            'token_data': token,
            'report_summary': {
                'name': token.get('name', ''),
                'symbol': token.get('symbol', ''),
                'price_usd': price_usd,
                'market_cap': price_usd * token.get('total_supply', 0),
                'liquidity_usd': liquidity_usd,
                'volume_24h': token.get('volume_24h', 0),
                'holders': holders,
                'pairs_count': len(pairs),
                'verified_contract': token.get('verified', False),
                'age_days': self._calculate_token_age(token.get('created_at', 'unknown')),
                'risk_score': risk_score,
                'risk_category': risk_category,
                'overview': overview
            },
            'risk_factors': {
                'contract_risk': token.get('contract_risk_level', 'unknown'),
                'risky_functions': token.get('risky_functions', []),
                'similar_to_scam': token.get('contract_similar_to_scam', False),
                'liquidity_risk': 'High' if liquidity_usd < 10000 else 'Medium' if liquidity_usd < 50000 else 'Low',
                'holders_risk': 'High' if holders < 50 else 'Medium' if holders < 200 else 'Low'
            },
            'recommendations': self._generate_recommendations(token),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_token_age(self, created_at: str) -> int:
        """
        Berechnet das Alter eines Tokens in Tagen.
        
        Args:
            created_at: Erstellungszeitpunkt als ISO-String
            
        Returns:
            Alter in Tagen
        """
        if created_at == 'unknown':
            return 0
        
        try:
            creation_time = datetime.fromisoformat(created_at)
            days_since_creation = (datetime.now() - creation_time).total_seconds() / (24 * 3600)
            return int(days_since_creation)
        except:
            return 0
    
    def _generate_recommendations(self, token: Dict[str, Any]) -> List[str]:
        """
        Generiert Handlungsempfehlungen basierend auf Token-Daten.
        
        Args:
            token: Token-Daten
            
        Returns:
            Liste von Empfehlungen
        """
        recommendations = []
        
        # Risiko-basierte Empfehlungen
        risk_score = token.get('risk_score', 0)
        
        if risk_score > 80:
            recommendations.append("AVOID: This token shows extremely high risk indicators.")
        elif risk_score > 60:
            recommendations.append("CAUTION: Only consider with extreme caution and thorough research.")
        
        # Kontraktbasierte Empfehlungen
        if not token.get('verified', False):
            recommendations.append("The contract is not verified - this is a major red flag.")
        
        if token.get('contract_similar_to_scam', False):
            recommendations.append("The contract code appears similar to known scam patterns.")
        
        # Riskante Funktionen
        risky_functions = token.get('risky_functions', [])
        if 'mint' in risky_functions:
            recommendations.append("The contract has minting capabilities which could lead to supply inflation.")
        
        if 'blacklist' in risky_functions:
            recommendations.append("The contract contains blacklist functionality which could restrict trading.")
        
        if 'fee_manipulation' in risky_functions:
            recommendations.append("The contract allows fee manipulation which could lead to unexpected costs.")
        
        # Liquiditätsbasierte Empfehlungen
        liquidity = token.get('liquidity_usd', 0)
        if liquidity < 5000:
            recommendations.append("The token has very low liquidity which could lead to high slippage and price manipulation.")
        elif liquidity < 20000:
            recommendations.append("The token has moderate liquidity - be cautious about position sizes to avoid slippage.")
        
        # Halterbasierte Empfehlungen
        holders = token.get('holders', 0)
        if holders < 20:
            recommendations.append("Very few holders could indicate a concentrated ownership which is a risk factor.")
        
        # Standardempfehlungen, wenn nichts Spezifisches
        if not recommendations:
            if risk_score < 30:
                recommendations.append("This token appears to have a reasonable risk profile, but always conduct your own research.")
            else:
                recommendations.append("Consider starting with a small position and monitoring the token's performance closely.")
        
        return recommendations

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Konfiguration
    config = {
        'etherscan_key': os.getenv('ETHERSCAN_API_KEY', ''),
        'bscscan_key': os.getenv('BSCSCAN_API_KEY', ''),
        'eth_rpc_url': os.getenv('ETH_RPC_URL', 'https://mainnet.infura.io/v3/your-infura-key'),
        'bsc_rpc_url': os.getenv('BSC_RPC_URL', 'https://bsc-dataseed.binance.org/')
    }
    
    # TokenScanner initialisieren
    scanner = TokenScanner(config)
    
    # Scanning nach neuen Token starten
    new_tokens = scanner.scan_new_tokens()
    print(f"Neue Token gefunden: {len(new_tokens)}")
    
    # Potenzielle Gems finden
    gems = scanner.get_potential_gems(max_tokens=5)
    print(f"Potenzielle Gems: {len(gems)}")
    
    for gem in gems:
        print(f"Symbol: {gem.get('symbol')}, Chain: {gem.get('chain')}, Score: {gem.get('gem_score')}")
    
    # Kontinuierliches Scanning starten
    # scanner.start_scanning(interval_seconds=300)

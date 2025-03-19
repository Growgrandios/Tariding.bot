import ipaddress
import logging
import json
import os
from functools import wraps
import time
import hmac
import hashlib
import base64
import threading
from google.cloud import secretmanager  # Neue Abhängigkeit für Secret Manager

class SecurityManager:
    def __init__(self, config_file="data/config/security.json", 
                 use_cloud_secrets=False, gcp_project=None):
        self.logger = logging.getLogger("security")
        self.config_file = config_file
        
        # Cloud Secret Manager Konfiguration
        self.use_cloud_secrets = use_cloud_secrets
        self.gcp_project = gcp_project
        self.secret_client = None
        
        if self.use_cloud_secrets and self.gcp_project:
            try:
                self.secret_client = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                self.logger.error(f"Fehler bei der Initialisierung des Secret Manager Clients: {str(e)}")
        
        # Standardkonfiguration
        self.default_config = {
            "allowed_ips": [],
            "rate_limit": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            },
            "failed_login_threshold": 5,
            "lockout_duration_minutes": 30,
            "ip_cache_minutes": 10  # Neuer Parameter: wie lange IP-Validierungen gecacht werden
        }
        
        self.config = self.load_config()
        
        # Rate-Limiting-Zähler
        self.request_counters = {}
        self.failed_login_attempts = {}
        self.locked_out_ips = {}
        
        # Cache für IP-Validierung
        self.ip_validation_cache = {}
        
        # API-Secrets aus sicherer Quelle laden
        self.api_secrets = self._load_api_secrets()
        
        # Bereinigungs-Thread für veraltete Einträge starten
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_entries, daemon=True)
        self.cleanup_thread.start()
    
    def _load_api_secrets(self):
        """Lädt API-Secrets aus sicherer Quelle"""
        secrets = {}
        
        # Versuche zuerst, aus Cloud Secret Manager zu laden
        if self.use_cloud_secrets and self.secret_client:
            try:
                # API-Secret aus Secret Manager laden
                name = f"projects/{self.gcp_project}/secrets/api_secrets/versions/latest"
                response = self.secret_client.access_secret_version(request={"name": name})
                secrets_data = response.payload.data.decode("UTF-8")
                secrets = json.loads(secrets_data)
                self.logger.info("API-Secrets aus Cloud Secret Manager geladen")
                return secrets
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der API-Secrets aus dem Secret Manager: {str(e)}")
        
        # Fallback: Credentials-Manager verwenden
        try:
            from src.utils.encryption import CredentialManager
            cred_manager = CredentialManager()
            credentials = cred_manager.decrypt_credentials()
            
            if credentials:
                # Beispiel für API-Secret-Extraktion
                if "API_SECRET" in credentials:
                    secrets["api_secret"] = credentials["API_SECRET"]
                elif "BITGET_API_SECRET" in credentials:
                    secrets["api_secret"] = credentials["BITGET_API_SECRET"]
            
            return secrets
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der API-Secrets aus dem Credential-Manager: {str(e)}")
        
        self.logger.warning("Keine API-Secrets geladen. Die Signaturvalidierung funktioniert nicht richtig.")
        return {"api_secret": "UNSECURE_PLACEHOLDER"}
    
    def load_config(self):
        """Lädt die Sicherheitskonfiguration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der Sicherheitskonfiguration: {str(e)}")
                return self.default_config
        else:
            # Standardkonfiguration speichern
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            return self.default_config
    
    def save_config(self):
        """Speichert die Sicherheitskonfiguration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Sicherheitskonfiguration: {str(e)}")
            return False
    
    def is_ip_allowed(self, ip_address):
        """Überprüft, ob eine IP-Adresse erlaubt ist (mit Caching)"""
        current_time = time.time()
        
        # Wenn keine IP-Beschränkungen konfiguriert sind, alle zulassen
        if not self.config.get("allowed_ips"):
            return True
        
        # Cache prüfen
        if ip_address in self.ip_validation_cache:
            cache_result, cache_expiry = self.ip_validation_cache[ip_address]
            if current_time < cache_expiry:
                return cache_result
        
        # Prüfen, ob die IP in der zulässigen Liste ist
        result = False
        for allowed_ip in self.config["allowed_ips"]:
            # IP-Bereiche unterstützen (CIDR-Notation)
            if "/" in allowed_ip:
                try:
                    network = ipaddress.ip_network(allowed_ip)
                    if ipaddress.ip_address(ip_address) in network:
                        result = True
                        break
                except ValueError:
                    continue
            # Einzelne IP-Adresse
            elif allowed_ip == ip_address:
                result = True
                break
        
        # Ergebnis cachen
        cache_expiry = current_time + (self.config.get("ip_cache_minutes", 10) * 60)
        self.ip_validation_cache[ip_address] = (result, cache_expiry)
        
        return result
    
    def add_allowed_ip(self, ip_address):
        """Fügt eine erlaubte IP-Adresse hinzu"""
        if ip_address not in self.config["allowed_ips"]:
            self.config["allowed_ips"].append(ip_address)
            self.save_config()
            
            # Cache aktualisieren
            current_time = time.time()
            cache_expiry = current_time + (self.config.get("ip_cache_minutes", 10) * 60)
            self.ip_validation_cache[ip_address] = (True, cache_expiry)
            
            self.logger.info(f"IP-Adresse {ip_address} zur Zulassungsliste hinzugefügt")
            return True
        return False
    
    def remove_allowed_ip(self, ip_address):
        """Entfernt eine erlaubte IP-Adresse"""
        if ip_address in self.config["allowed_ips"]:
            self.config["allowed_ips"].remove(ip_address)
            self.save_config()
            
            # Cache aktualisieren, falls vorhanden
            if ip_address in self.ip_validation_cache:
                current_time = time.time()
                cache_expiry = current_time + (self.config.get("ip_cache_minutes", 10) * 60)
                self.ip_validation_cache[ip_address] = (False, cache_expiry)
            
            self.logger.info(f"IP-Adresse {ip_address} von der Zulassungsliste entfernt")
            return True
        return False
    
    def check_rate_limit(self, ip_address):
        """Überprüft, ob eine IP-Adresse das Rate-Limit überschritten hat"""
        current_time = time.time()
        
        # Counter initialisieren, falls noch nicht vorhanden
        if ip_address not in self.request_counters:
            self.request_counters[ip_address] = {
                "minute": {"count": 0, "reset_time": current_time + 60},
                "hour": {"count": 0, "reset_time": current_time + 3600}
            }
        
        # Zähler zurücksetzen, falls die Zeit abgelaufen ist
        for period in ["minute", "hour"]:
            if current_time > self.request_counters[ip_address][period]["reset_time"]:
                self.request_counters[ip_address][period]["count"] = 0
                self.request_counters[ip_address][period]["reset_time"] = current_time + (60 if period == "minute" else 3600)
        
        # Zähler erhöhen
        self.request_counters[ip_address]["minute"]["count"] += 1
        self.request_counters[ip_address]["hour"]["count"] += 1
        
        # Prüfen, ob das Limit überschritten wurde
        if self.request_counters[ip_address]["minute"]["count"] > self.config["rate_limit"]["requests_per_minute"]:
            self.logger.warning(f"Rate-Limit pro Minute für IP {ip_address} überschritten")
            return False
        
        if self.request_counters[ip_address]["hour"]["count"] > self.config["rate_limit"]["requests_per_hour"]:
            self.logger.warning(f"Rate-Limit pro Stunde für IP {ip_address} überschritten")
            return False
        
        return True
    
    def validate_api_signature(self, api_key, signature, timestamp, body):
        """Überprüft die Signatur einer API-Anfrage"""
        if not self.api_secrets.get("api_secret"):
            self.logger.error("Kein API-Secret verfügbar für Signaturvalidierung")
            return False
        
        stored_secret = self.api_secrets.get("api_secret")
        
        # Message erstellen
        message = timestamp + api_key + body
        
        # Signatur erstellen
        expected_signature = base64.b64encode(
            hmac.new(
                stored_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        # Signatur vergleichen
        return hmac.compare_digest(signature, expected_signature)
    
    def record_failed_login(self, ip_address):
        """Zeichnet einen fehlgeschlagenen Anmeldeversuch auf"""
        current_time = time.time()
        
        # Prüfen, ob die IP gesperrt ist
        if ip_address in self.locked_out_ips:
            if current_time < self.locked_out_ips[ip_address]:
                # IP ist noch gesperrt
                return False
            else:
                # Sperrzeit abgelaufen, Zähler zurücksetzen
                del self.locked_out_ips[ip_address]
                self.failed_login_attempts[ip_address] = 0
        
        # Fehlversuch zählen
        if ip_address not in self.failed_login_attempts:
            self.failed_login_attempts[ip_address] = 1
        else:
            self.failed_login_attempts[ip_address] += 1
        
        # Prüfen, ob der Schwellenwert überschritten wurde
        if self.failed_login_attempts[ip_address] >= self.config["failed_login_threshold"]:
            # IP für die konfigurierte Zeit sperren
            lockout_time = current_time + (self.config["lockout_duration_minutes"] * 60)
            self.locked_out_ips[ip_address] = lockout_time
            self.logger.warning(f"IP {ip_address} wurde wegen zu vieler fehlgeschlagener Anmeldeversuche gesperrt")
            return False
        
        return True
    
    def is_ip_locked_out(self, ip_address):
        """Überprüft, ob eine IP gesperrt ist"""
        current_time = time.time()
        
        if ip_address in self.locked_out_ips:
            if current_time < self.locked_out_ips[ip_address]:
                # IP ist gesperrt
                return True
            else:
                # Sperrzeit abgelaufen
                del self.locked_out_ips[ip_address]
                self.failed_login_attempts[ip_address] = 0
        
        return False
    
    def _cleanup_expired_entries(self):
        """Bereinigt veraltete Einträge in regelmäßigen Abständen"""
        while True:
            try:
                current_time = time.time()
                
                # Veraltete IP-Validierungscache-Einträge bereinigen
                for ip in list(self.ip_validation_cache.keys()):
                    _, expiry_time = self.ip_validation_cache[ip]
                    if current_time > expiry_time:
                        del self.ip_validation_cache[ip]
                
                # Veraltete Rate-Limit-Einträge bereinigen
                for ip in list(self.request_counters.keys()):
                    # Wenn beide Perioden abgelaufen sind, Eintrag entfernen
                    if (current_time > self.request_counters[ip]["minute"]["reset_time"] and 
                        current_time > self.request_counters[ip]["hour"]["reset_time"]):
                        del self.request_counters[ip]
                
                # Veraltete Lockout-Einträge bereinigen
                for ip in list(self.locked_out_ips.keys()):
                    if current_time > self.locked_out_ips[ip]:
                        del self.locked_out_ips[ip]
                        if ip in self.failed_login_attempts:
                            del self.failed_login_attempts[ip]
                
                # Alle 5 Minuten ausführen
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"Fehler bei der Bereinigung veralteter Einträge: {str(e)}")
                time.sleep(60)  # Bei Fehler kürzere Wartezeit

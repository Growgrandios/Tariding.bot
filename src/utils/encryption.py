from cryptography.fernet import Fernet
import os
import base64
import getpass
import json
import datetime
import logging
from google.cloud import secretmanager  # Neue Abhängigkeit für Google Secret Manager

class CredentialManager:
    def __init__(self, key_file='.key', encrypted_file='.env.encrypted', 
                 use_cloud_secrets=False, gcp_project=None):
        self.key_file = key_file
        self.encrypted_file = encrypted_file
        self.key = None
        self.cipher = None
        self.logger = logging.getLogger("credentials")
        
        # Cloud Secret Manager Konfiguration
        self.use_cloud_secrets = use_cloud_secrets
        self.gcp_project = gcp_project
        self.client = None
        
        if self.use_cloud_secrets and self.gcp_project:
            try:
                self.client = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                self.logger.error(f"Fehler bei der Initialisierung des Secret Manager Clients: {str(e)}")
    
    def generate_key(self):
        """Generiert einen neuen Verschlüsselungsschlüssel"""
        key = Fernet.generate_key()
        
        if self.use_cloud_secrets and self.client:
            # Schlüssel in Cloud Secret Manager speichern
            try:
                parent = f"projects/{self.gcp_project}"
                secret_id = "gemmatrader_encryption_key"
                
                # Prüfen, ob Secret bereits existiert
                try:
                    self.client.get_secret(name=f"{parent}/secrets/{secret_id}")
                except Exception:
                    # Secret erstellen
                    secret = {"replication": {"automatic": {}}}
                    self.client.create_secret(
                        request={"parent": parent, "secret_id": secret_id, "secret": secret}
                    )
                
                # Neue Version des Secrets erstellen
                self.client.add_secret_version(
                    request={"parent": f"{parent}/secrets/{secret_id}", "payload": {"data": key}}
                )
                self.logger.info("Verschlüsselungsschlüssel im Cloud Secret Manager gespeichert")
            except Exception as e:
                self.logger.error(f"Fehler beim Speichern des Schlüssels im Secret Manager: {str(e)}")
                # Fallback: lokale Speicherung
                self._store_key_locally(key)
        else:
            # Lokale Speicherung
            self._store_key_locally(key)
        
        return key
    
    def _store_key_locally(self, key):
        """Speichert den Schlüssel lokal mit entsprechenden Berechtigungen"""
        with open(self.key_file, 'wb') as f:
            f.write(key)
        os.chmod(self.key_file, 0o600)  # Nur für Besitzer lesbar
    
    def load_key(self):
        """Lädt den Verschlüsselungsschlüssel"""
        if self.use_cloud_secrets and self.client:
            try:
                name = f"projects/{self.gcp_project}/secrets/gemmatrader_encryption_key/versions/latest"
                response = self.client.access_secret_version(request={"name": name})
                return response.payload.data
            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Schlüssels aus dem Secret Manager: {str(e)}")
                # Fallback: versuche lokale Datei
        
        # Lokale Datei verwenden
        if not os.path.exists(self.key_file):
            return self.generate_key()
        
        with open(self.key_file, 'rb') as f:
            return f.read()
    
    def initialize(self):
        """Initialisiert den Verschlüsselungsmanager"""
        self.key = self.load_key()
        self.cipher = Fernet(self.key)
    
    def rotate_key(self):
        """Rotiert den Verschlüsselungsschlüssel und verschlüsselt Daten neu"""
        # Aktuelle Credentials sichern
        current_credentials = self.decrypt_credentials()
        if not current_credentials:
            self.logger.error("Keine Credentials zum Neu-Verschlüsseln gefunden")
            return False
        
        # Backup des alten Schlüssels erstellen
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(self.key_file):
            backup_key_file = f"{self.key_file}_{now}.bak"
            os.rename(self.key_file, backup_key_file)
            self.logger.info(f"Backup des alten Schlüssels erstellt: {backup_key_file}")
        
        # Neuen Schlüssel generieren und Credentials neu verschlüsseln
        self.key = self.generate_key()
        self.cipher = Fernet(self.key)
        success = self.encrypt_credentials(current_credentials)
        
        if success:
            self.logger.info("Schlüssel erfolgreich rotiert und Credentials neu verschlüsselt")
            return True
        else:
            self.logger.error("Fehler bei der Schlüsselrotation")
            return False
    
    def encrypt_credentials(self, credentials_dict):
        """Verschlüsselt Zugangsdaten"""
        if not self.cipher:
            self.initialize()
        
        # Konvertiere Dict zu JSON-String
        credentials_json = json.dumps(credentials_dict).encode()
        
        # Verschlüsseln
        encrypted_data = self.cipher.encrypt(credentials_json)
        
        # Speichern
        with open(self.encrypted_file, 'wb') as f:
            f.write(encrypted_data)
        os.chmod(self.encrypted_file, 0o600)  # Nur für Besitzer lesbar
        
        # In Cloud Secret Manager speichern, falls aktiviert
        if self.use_cloud_secrets and self.client:
            try:
                parent = f"projects/{self.gcp_project}"
                secret_id = "gemmatrader_credentials"
                
                # Prüfen, ob Secret bereits existiert
                try:
                    self.client.get_secret(name=f"{parent}/secrets/{secret_id}")
                except Exception:
                    # Secret erstellen
                    secret = {"replication": {"automatic": {}}}
                    self.client.create_secret(
                        request={"parent": parent, "secret_id": secret_id, "secret": secret}
                    )
                
                # Neue Version des Secrets erstellen
                self.client.add_secret_version(
                    request={
                        "parent": f"{parent}/secrets/{secret_id}", 
                        "payload": {"data": encrypted_data}
                    }
                )
                self.logger.info("Verschlüsselte Credentials im Cloud Secret Manager gespeichert")
            except Exception as e:
                self.logger.error(f"Fehler beim Speichern der Credentials im Secret Manager: {str(e)}")
        
        return True
    
    def decrypt_credentials(self):
        """Entschlüsselt Zugangsdaten"""
        if not self.cipher:
            self.initialize()
        
        encrypted_data = None
        
        # Versuche zuerst, aus Cloud Secret Manager zu laden
        if self.use_cloud_secrets and self.client:
            try:
                name = f"projects/{self.gcp_project}/secrets/gemmatrader_credentials/versions/latest"
                response = self.client.access_secret_version(request={"name": name})
                encrypted_data = response.payload.data
                self.logger.info("Credentials aus Cloud Secret Manager geladen")
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der Credentials aus dem Secret Manager: {str(e)}")
        
        # Fallback: Lokale Datei
        if not encrypted_data and os.path.exists(self.encrypted_file):
            with open(self.encrypted_file, 'rb') as f:
                encrypted_data = f.read()
        
        if not encrypted_data:
            return None
        
        # Entschlüsseln
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            # JSON-String zu Dict konvertieren
            credentials_dict = json.loads(decrypted_data)
            return credentials_dict
        except Exception as e:
            self.logger.error(f"Fehler beim Entschlüsseln der Credentials: {str(e)}")
            return None
    
    def update_credential(self, key, value):
        """Aktualisiert einen einzelnen Credential-Wert"""
        credentials = self.decrypt_credentials() or {}
        credentials[key] = value
        return self.encrypt_credentials(credentials)
    
    def setup_interactive(self):
        """Führt interaktives Setup für Zugangsdaten durch"""
        print("=== Sichere Konfiguration von API-Schlüsseln ===")
        print("Hinweis: Die Schlüssel werden verschlüsselt gespeichert.")
        
        credentials = self.decrypt_credentials() or {}
        
        # Bitget API-Zugangsdaten
        credentials['BITGET_API_KEY'] = getpass.getpass("Bitget API Key: ")
        credentials['BITGET_API_SECRET'] = getpass.getpass("Bitget API Secret: ")
        credentials['BITGET_API_PASSPHRASE'] = getpass.getpass("Bitget API Passphrase: ")
        
        # Telegram-Bot-Token
        credentials['TELEGRAM_BOT_TOKEN'] = getpass.getpass("Telegram Bot Token: ")
        
        # Alpha Vantage API-Key (optional)
        use_alpha_vantage = input("Alpha Vantage API-Key konfigurieren? (j/n): ").lower() == 'j'
        if use_alpha_vantage:
            credentials['ALPHA_VANTAGE_API_KEY'] = getpass.getpass("Alpha Vantage API Key: ")
        
        success = self.encrypt_credentials(credentials)
        
        if success:
            print("Zugangsdaten wurden sicher gespeichert.")
        else:
            print("Fehler beim Speichern der Zugangsdaten.")

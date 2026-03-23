import os
import logging
from typing import Optional
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseStorageManager:
    """Upload manager for Firebase Storage (optional)."""
    
    def __init__(self):
        load_dotenv()
        self.enabled = False

        # Initialize Firebase only if configuration is present
        if not firebase_admin._apps:
            self._initialize_firebase()

        if firebase_admin._apps:
            self.bucket = storage.bucket()
            self.enabled = True
        else:
            self.bucket = None
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK if configuration is available."""
        try:
            # Build credentials from environment variables
            cred_dict = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or "").replace("\\n", "\n"),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
                "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
                "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
            }

            # If any critical field is missing, do not enable Firebase
            required_keys = [
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri",
                "auth_provider_x509_cert_url",
                "client_x509_cert_url",
            ]
            if any(not cred_dict[k] for k in required_keys):
                logger.warning(
                    "Firebase configuration not found or incomplete. "
                    "Firebase Storage integration will be disabled."
                )
                return
            
            cred = credentials.Certificate(cred_dict)
            
            # Prefer explicit env bucket; fallback to project-based default.
            bucket_name = (
                os.getenv("FIREBASE_STORAGE_BUCKET")
                or f"{cred_dict['project_id']}.firebasestorage.app"
            )
            
            firebase_admin.initialize_app(
                cred,
                {
                    "storageBucket": bucket_name,
                },
            )
            
            logger.info(f"Firebase Admin SDK initialized successfully - Bucket: {bucket_name}")
            
        except Exception as e:
            logger.warning(f"Could not initialize Firebase Storage (will be disabled): {e}")
    
    def upload_file(self, file_content: bytes, file_name: str, document_id: int) -> Optional[str]:
        """Upload a file to Firebase Storage (if enabled)."""
        if not self.enabled or not self.bucket:
            logger.info(
                "Firebase Storage is not enabled. Skipping file upload for "
                f"{file_name} (document_id={document_id})."
            )
            return None

        try:
            # Build a deterministic path for the file
            file_extension = os.path.splitext(file_name)[1]
            unique_filename = f"documents/{document_id}/{file_name}"
            
            blob = self.bucket.blob(unique_filename)
            blob.upload_from_string(
                file_content,
                content_type=self._get_content_type(file_extension),
            )
            blob.make_public()
            public_url = blob.public_url
            logger.info(f"File {file_name} uploaded to Firebase Storage: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"Error uploading file to Firebase Storage: {e}")
            return None
    

    
    def delete_file(self, file_url: str) -> bool:
        """Delete a file from Firebase Storage."""
        if not self.enabled or not self.bucket:
            logger.info(
                "Firebase Storage is not enabled. Skipping file deletion for URL: %s",
                file_url,
            )
            return False
        try:
            # Extract file path from URL
            file_path = self._extract_file_path_from_url(file_url)
            
            if file_path:
                blob = self.bucket.blob(file_path)
                blob.delete()
                logger.info(f"File deleted from Firebase Storage: {file_path}")
                return True
            else:
                logger.warning(f"Could not extract file path from URL: {file_url}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file from Firebase Storage: {e}")
            return False
    

    
    def _get_content_type(self, file_extension: str) -> str:
        """Get content-type based on file extension."""
        content_types = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.rtf': 'application/rtf'
        }
        
        return content_types.get(file_extension.lower(), 'application/octet-stream')
    
    def _extract_file_path_from_url(self, url: str) -> Optional[str]:
        """Extract file path from a Firebase Storage public URL."""
        try:
            # Firebase Storage URL pattern: https://storage.googleapis.com/BUCKET_NAME/path/to/file
            parts = url.split('/')
            if len(parts) >= 5:
                # Pular 'https:', '', 'storage.googleapis.com', 'BUCKET_NAME'
                file_path = '/'.join(parts[4:])
                return file_path
            return None
        except Exception as e:
            logger.error(f"Error extracting file path from URL: {e}")
            return None

# Global instance
firebase_storage = FirebaseStorageManager() 
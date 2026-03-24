import os
import logging
from typing import Optional
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirebaseStorageManager:
    """Upload manager for Firebase Storage (optional)."""

    def __init__(self):
        load_dotenv()
        self.enabled = False

        if not firebase_admin._apps:
            self._initialize_firebase()

        if firebase_admin._apps:
            self.bucket = storage.bucket()
            self.enabled = True
        else:
            self.bucket = None

    def _initialize_firebase(self):
        try:
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

            required_keys = [
                "project_id", "private_key_id", "private_key", "client_email", "client_id",
                "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url",
            ]
            if any(not cred_dict[k] for k in required_keys):
                logger.warning(
                    "Firebase configuration not found or incomplete. "
                    "Firebase Storage integration will be disabled."
                )
                return

            cred = credentials.Certificate(cred_dict)
            bucket_name = (
                os.getenv("FIREBASE_STORAGE_BUCKET")
                or f"{cred_dict['project_id']}.firebasestorage.app"
            )
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
            logger.info("Firebase Admin SDK initialized successfully - Bucket: %s", bucket_name)

        except Exception as e:
            logger.warning("Could not initialize Firebase Storage (will be disabled): %s", e)

    def upload_file(self, file_content: bytes, file_name: str, document_id: int) -> Optional[str]:
        if not self.enabled or not self.bucket:
            logger.info(
                "Firebase Storage is not enabled. Skipping file upload for %s (document_id=%s).",
                file_name, document_id,
            )
            return None

        try:
            unique_filename = f"documents/{document_id}/{file_name}"
            file_extension = os.path.splitext(file_name)[1]
            blob = self.bucket.blob(unique_filename)
            blob.upload_from_string(
                file_content,
                content_type=self._get_content_type(file_extension),
            )
            blob.make_public()
            public_url = blob.public_url
            logger.info("File %s uploaded to Firebase Storage: %s", file_name, public_url)
            return public_url
        except Exception as e:
            logger.error("Error uploading file to Firebase Storage: %s", e)
            return None

    def delete_file(self, file_url: str) -> bool:
        if not self.enabled or not self.bucket:
            logger.info("Firebase Storage is not enabled. Skipping file deletion for URL: %s", file_url)
            return False
        try:
            file_path = self._extract_file_path_from_url(file_url)
            if file_path:
                blob = self.bucket.blob(file_path)
                blob.delete()
                logger.info("File deleted from Firebase Storage: %s", file_path)
                return True
            logger.warning("Could not extract file path from URL: %s", file_url)
            return False
        except Exception as e:
            logger.error("Error deleting file from Firebase Storage: %s", e)
            return False

    def _get_content_type(self, file_extension: str) -> str:
        content_types = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.rtf': 'application/rtf'
        }
        return content_types.get(file_extension.lower(), 'application/octet-stream')

    def _extract_file_path_from_url(self, url: str) -> Optional[str]:
        try:
            parts = url.split('/')
            if len(parts) >= 5:
                return '/'.join(parts[4:])
            return None
        except Exception as e:
            logger.error("Error extracting file path from URL: %s", e)
            return None


firebase_storage = FirebaseStorageManager()

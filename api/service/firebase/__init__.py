__all__ = [
    "send_notification",
]

from pathlib import Path

from firebase_admin import initialize_app
from firebase_admin.credentials import Certificate

ROOT = Path(__file__).parent

initialize_app(Certificate(ROOT/"service_account.json"))

from .notification import send_notification

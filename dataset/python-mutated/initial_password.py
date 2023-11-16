import base64
import hashlib
from typing import Optional
from django.conf import settings

def initial_password(email: str) -> Optional[str]:
    if False:
        return 10
    'Given an email address, returns the initial password for that account, as\n    created by populate_db.'
    if settings.INITIAL_PASSWORD_SALT is not None:
        assert settings.DEVELOPMENT, 'initial_password_salt should not be set in production.'
        encoded_key = (settings.INITIAL_PASSWORD_SALT + email).encode()
        digest = hashlib.sha256(encoded_key).digest()
        return base64.b64encode(digest)[:16].decode()
    else:
        return None
"""Provides cloud secrets services."""
from __future__ import annotations
import functools
from core import feconf
from core.constants import constants
from google import auth
from google.cloud import secretmanager
from typing import Optional
CLIENT = secretmanager.SecretManagerServiceClient(credentials=auth.credentials.AnonymousCredentials() if constants.EMULATOR_MODE else auth.default()[0])

@functools.lru_cache(maxsize=64)
def get_secret(name: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Gets the value of a secret.\n\n    Args:\n        name: str. The name of the secret to retrieve.\n\n    Returns:\n        str. The value of the secret.\n    '
    secret_name = f'projects/{feconf.OPPIA_PROJECT_ID}/secrets/{name}/versions/latest'
    try:
        response = CLIENT.access_secret_version(request={'name': secret_name})
    except Exception:
        return None
    return response.payload.data.decode('utf-8')
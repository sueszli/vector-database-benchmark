"""Provides dev mode secrets services."""
from __future__ import annotations
import functools
import json
import os
from typing import Optional

@functools.lru_cache(maxsize=64)
def get_secret(name: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Gets the value of a secret. This is only dev mode version of the secrets.\n\n    Args:\n        name: str. The name of the secret to retrieve.\n\n    Returns:\n        str. The value of the secret.\n    '
    secret = json.loads(os.environ.get('SECRETS', '{}')).get(name)
    assert secret is None or isinstance(secret, str)
    return secret
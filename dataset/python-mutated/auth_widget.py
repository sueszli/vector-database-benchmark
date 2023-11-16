import hashlib
import hmac
from typing import Any, Dict

def check_signature(token: str, hash: str, **kwargs: Any) -> bool:
    if False:
        print('Hello World!')
    "\n    Generate hexadecimal representation\n    of the HMAC-SHA-256 signature of the data-check-string\n    with the SHA256 hash of the bot's token used as a secret key\n\n    :param token:\n    :param hash:\n    :param kwargs: all params received on auth\n    :return:\n    "
    secret = hashlib.sha256(token.encode('utf-8'))
    check_string = '\n'.join((f'{k}={kwargs[k]}' for k in sorted(kwargs)))
    hmac_string = hmac.new(secret.digest(), check_string.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
    return hmac_string == hash

def check_integrity(token: str, data: Dict[str, Any]) -> bool:
    if False:
        while True:
            i = 10
    "\n    Verify the authentication and the integrity\n    of the data received on user's auth\n\n    :param token: Bot's token\n    :param data: all data that came on auth\n    :return:\n    "
    return check_signature(token, **data)
import hashlib
import hmac
from django.conf import settings

def generate_camo_url(url: str) -> str:
    if False:
        i = 10
        return i + 15
    encoded_url = url.encode()
    assert settings.CAMO_KEY is not None
    encoded_camo_key = settings.CAMO_KEY.encode()
    digest = hmac.new(encoded_camo_key, encoded_url, hashlib.sha1).hexdigest()
    return f'{digest}/{encoded_url.hex()}'

def get_camo_url(url: str) -> str:
    if False:
        while True:
            i = 10
    if settings.CAMO_URI == '':
        return url
    return f'{settings.CAMO_URI}{generate_camo_url(url)}'

def is_camo_url_valid(digest: str, url: str) -> bool:
    if False:
        print('Hello World!')
    camo_url = generate_camo_url(url)
    camo_url_digest = camo_url.split('/')[0]
    return camo_url_digest == digest
import hashlib
import urllib.parse

def _hash(email: str | None) -> str:
    if False:
        return 10
    if email is None:
        email = ''
    return hashlib.md5(email.strip().lower().encode('utf8'), usedforsecurity=False).hexdigest()

def gravatar(request, email, size=80):
    if False:
        for i in range(10):
            print('nop')
    url = f'https://secure.gravatar.com/avatar/{_hash(email)}'
    params = {'size': size}
    return request.camo_url('?'.join([url, urllib.parse.urlencode(params)]))

def profile(email: str) -> str:
    if False:
        while True:
            i = 10
    return f'https://gravatar.com/{_hash(email)}'
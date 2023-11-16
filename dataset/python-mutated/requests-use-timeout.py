import requests
url = 'www.github.com'
r = requests.get(url)
r = requests.post(url)
r = requests.request('GET', url)

def return_url():
    if False:
        while True:
            i = 10
    return url
r = requests.request('GET', return_url())
r = requests.get(url, timeout=50)

def from_import_test1(url):
    if False:
        return 10
    from requests import get, post
    r = get(url, timeout=3)
    r = post(url)

def test2():
    if False:
        i = 10
        return i + 15
    'Perform a requests.get and default headers set'
    headers = {**_get_default_headers(), **headers}
    r = requests.get(url, headers=headers, params=params, **{'timeout': TIMEOUT, **kwargs})
    return r
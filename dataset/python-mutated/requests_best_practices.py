from ray import serve

@serve.deployment
def f(*args):
    if False:
        for i in range(10):
            print('nop')
    return 'Hi there!'
serve.run(f.bind())
import requests
response = requests.get('http://localhost:8000/')
result = response.text
assert result == 'Hi there!'
import requests
from requests.adapters import HTTPAdapter, Retry
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 501, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
response = session.get('http://localhost:8000/', timeout=10)
result = response.text
assert result == 'Hi there!'
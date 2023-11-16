import os
import uuid
import requests

def test_no_args():
    if False:
        print('Hello World!')
    BASE_URL = os.getenv('BASE_URL')
    assert BASE_URL is not None
    res = requests.get(f'{BASE_URL}/hello_http')
    assert res.text == 'Hello, World!'

def test_args():
    if False:
        print('Hello World!')
    BASE_URL = os.getenv('BASE_URL')
    assert BASE_URL is not None
    name = str(uuid.uuid4())
    res = requests.post(f'{BASE_URL}/hello_http', json={'name': name})
    assert res.text == f'Hello, {name}!'
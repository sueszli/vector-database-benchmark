from functools import partial
from typing import Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient

def main(some_arg, q: Optional[str]=None):
    if False:
        print('Hello World!')
    return {'some_arg': some_arg, 'q': q}
endpoint = partial(main, 'foo')
app = FastAPI()
app.get('/')(endpoint)
client = TestClient(app)

def test_partial():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/?q=bar')
    data = response.json()
    assert data == {'some_arg': 'foo', 'q': 'bar'}
from typing import Optional
from fastapi import FastAPI
from fastapi.params import Param
from fastapi.testclient import TestClient
app = FastAPI()

@app.get('/items/')
def read_items(q: Optional[str]=Param(default=None)):
    if False:
        while True:
            i = 10
    return {'q': q}
client = TestClient(app)

def test_default_param_query_none():
    if False:
        i = 10
        return i + 15
    response = client.get('/items/')
    assert response.status_code == 200, response.text
    assert response.json() == {'q': None}

def test_default_param_query():
    if False:
        print('Hello World!')
    response = client.get('/items/?q=foo')
    assert response.status_code == 200, response.text
    assert response.json() == {'q': 'foo'}
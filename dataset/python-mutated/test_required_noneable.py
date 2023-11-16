from typing import Union
from fastapi import Body, FastAPI, Query
from fastapi.testclient import TestClient
app = FastAPI()

@app.get('/query')
def read_query(q: Union[str, None]):
    if False:
        print('Hello World!')
    return q

@app.get('/explicit-query')
def read_explicit_query(q: Union[str, None]=Query()):
    if False:
        print('Hello World!')
    return q

@app.post('/body-embed')
def send_body_embed(b: Union[str, None]=Body(embed=True)):
    if False:
        print('Hello World!')
    return b
client = TestClient(app)

def test_required_nonable_query_invalid():
    if False:
        print('Hello World!')
    response = client.get('/query')
    assert response.status_code == 422

def test_required_noneable_query_value():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/query', params={'q': 'foo'})
    assert response.status_code == 200
    assert response.json() == 'foo'

def test_required_nonable_explicit_query_invalid():
    if False:
        return 10
    response = client.get('/explicit-query')
    assert response.status_code == 422

def test_required_nonable_explicit_query_value():
    if False:
        return 10
    response = client.get('/explicit-query', params={'q': 'foo'})
    assert response.status_code == 200
    assert response.json() == 'foo'

def test_required_nonable_body_embed_no_content():
    if False:
        print('Hello World!')
    response = client.post('/body-embed')
    assert response.status_code == 422

def test_required_nonable_body_embed_invalid():
    if False:
        while True:
            i = 10
    response = client.post('/body-embed', json={'invalid': 'invalid'})
    assert response.status_code == 422

def test_required_noneable_body_embed_value():
    if False:
        i = 10
        return i + 15
    response = client.post('/body-embed', json={'b': 'foo'})
    assert response.status_code == 200
    assert response.json() == 'foo'
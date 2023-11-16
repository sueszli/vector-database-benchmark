from fastapi import FastAPI, Path, Query
from fastapi.testclient import TestClient
app = FastAPI()

@app.get('/int/{param:int}')
def int_convertor(param: int=Path()):
    if False:
        print('Hello World!')
    return {'int': param}

@app.get('/float/{param:float}')
def float_convertor(param: float=Path()):
    if False:
        for i in range(10):
            print('nop')
    return {'float': param}

@app.get('/path/{param:path}')
def path_convertor(param: str=Path()):
    if False:
        while True:
            i = 10
    return {'path': param}

@app.get('/query/')
def query_convertor(param: str=Query()):
    if False:
        print('Hello World!')
    return {'query': param}
client = TestClient(app)

def test_route_converters_int():
    if False:
        while True:
            i = 10
    response = client.get('/int/5')
    assert response.status_code == 200, response.text
    assert response.json() == {'int': 5}
    assert app.url_path_for('int_convertor', param=5) == '/int/5'

def test_route_converters_float():
    if False:
        return 10
    response = client.get('/float/25.5')
    assert response.status_code == 200, response.text
    assert response.json() == {'float': 25.5}
    assert app.url_path_for('float_convertor', param=25.5) == '/float/25.5'

def test_route_converters_path():
    if False:
        print('Hello World!')
    response = client.get('/path/some/example')
    assert response.status_code == 200, response.text
    assert response.json() == {'path': 'some/example'}

def test_route_converters_query():
    if False:
        return 10
    response = client.get('/query', params={'param': 'Qué tal!'})
    assert response.status_code == 200, response.text
    assert response.json() == {'query': 'Qué tal!'}

def test_url_path_for_path_convertor():
    if False:
        while True:
            i = 10
    assert app.url_path_for('path_convertor', param='some/example') == '/path/some/example'
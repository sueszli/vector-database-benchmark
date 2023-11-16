from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()

@app.get('/', openapi_extra={'x-custom-extension': 'value'})
def route_with_extras():
    if False:
        while True:
            i = 10
    return {}
client = TestClient(app)

def test_get_route():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/')
    assert response.status_code == 200, response.text
    assert response.json() == {}

def test_openapi():
    if False:
        return 10
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Route With Extras', 'operationId': 'route_with_extras__get', 'x-custom-extension': 'value'}}}}
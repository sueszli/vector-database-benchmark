from typing import Optional
from fastapi import FastAPI, Security
from fastapi.security import OAuth2AuthorizationCodeBearer
from fastapi.testclient import TestClient
app = FastAPI()
oauth2_scheme = OAuth2AuthorizationCodeBearer(authorizationUrl='authorize', tokenUrl='token', auto_error=True)

@app.get('/items/')
async def read_items(token: Optional[str]=Security(oauth2_scheme)):
    return {'token': token}
client = TestClient(app)

def test_no_token():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/items')
    assert response.status_code == 401, response.text
    assert response.json() == {'detail': 'Not authenticated'}

def test_incorrect_token():
    if False:
        i = 10
        return i + 15
    response = client.get('/items', headers={'Authorization': 'Non-existent testtoken'})
    assert response.status_code == 401, response.text
    assert response.json() == {'detail': 'Not authenticated'}

def test_token():
    if False:
        while True:
            i = 10
    response = client.get('/items', headers={'Authorization': 'Bearer testtoken'})
    assert response.status_code == 200, response.text
    assert response.json() == {'token': 'testtoken'}

def test_openapi_schema():
    if False:
        i = 10
        return i + 15
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/items/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Read Items', 'operationId': 'read_items_items__get', 'security': [{'OAuth2AuthorizationCodeBearer': []}]}}}, 'components': {'securitySchemes': {'OAuth2AuthorizationCodeBearer': {'type': 'oauth2', 'flows': {'authorizationCode': {'authorizationUrl': 'authorize', 'tokenUrl': 'token', 'scopes': {}}}}}}}
from typing import Optional
from fastapi import Depends, FastAPI, Security
from fastapi.security import APIKeyQuery
from fastapi.testclient import TestClient
from pydantic import BaseModel
app = FastAPI()
api_key = APIKeyQuery(name='key', auto_error=False)

class User(BaseModel):
    username: str

def get_current_user(oauth_header: Optional[str]=Security(api_key)):
    if False:
        while True:
            i = 10
    if oauth_header is None:
        return None
    user = User(username=oauth_header)
    return user

@app.get('/users/me')
def read_current_user(current_user: Optional[User]=Depends(get_current_user)):
    if False:
        i = 10
        return i + 15
    if current_user is None:
        return {'msg': 'Create an account first'}
    return current_user
client = TestClient(app)

def test_security_api_key():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/users/me?key=secret')
    assert response.status_code == 200, response.text
    assert response.json() == {'username': 'secret'}

def test_security_api_key_no_key():
    if False:
        return 10
    response = client.get('/users/me')
    assert response.status_code == 200, response.text
    assert response.json() == {'msg': 'Create an account first'}

def test_openapi_schema():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/users/me': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Read Current User', 'operationId': 'read_current_user_users_me_get', 'security': [{'APIKeyQuery': []}]}}}, 'components': {'securitySchemes': {'APIKeyQuery': {'type': 'apiKey', 'name': 'key', 'in': 'query'}}}}
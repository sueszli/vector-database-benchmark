from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

class ResponseModel(BaseModel):
    message: str
app = FastAPI()
router = APIRouter()

@router.get('/a', responses={501: {'description': 'Error 1'}})
async def a():
    return 'a'

@router.get('/b', responses={502: {'description': 'Error 2'}, '4XX': {'description': 'Error with range, upper'}})
async def b():
    return 'b'

@router.get('/c', responses={'400': {'description': 'Error with str'}, '5xx': {'description': 'Error with range, lower'}, 'default': {'description': 'A default response'}})
async def c():
    return 'c'

@router.get('/d', responses={'400': {'description': 'Error with str'}, '5XX': {'model': ResponseModel}, 'default': {'model': ResponseModel}})
async def d():
    return 'd'
app.include_router(router)
client = TestClient(app)

def test_a():
    if False:
        i = 10
        return i + 15
    response = client.get('/a')
    assert response.status_code == 200, response.text
    assert response.json() == 'a'

def test_b():
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/b')
    assert response.status_code == 200, response.text
    assert response.json() == 'b'

def test_c():
    if False:
        return 10
    response = client.get('/c')
    assert response.status_code == 200, response.text
    assert response.json() == 'c'

def test_d():
    if False:
        while True:
            i = 10
    response = client.get('/d')
    assert response.status_code == 200, response.text
    assert response.json() == 'd'

def test_openapi_schema():
    if False:
        i = 10
        return i + 15
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/a': {'get': {'responses': {'501': {'description': 'Error 1'}, '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'A', 'operationId': 'a_a_get'}}, '/b': {'get': {'responses': {'502': {'description': 'Error 2'}, '4XX': {'description': 'Error with range, upper'}, '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'B', 'operationId': 'b_b_get'}}, '/c': {'get': {'responses': {'400': {'description': 'Error with str'}, '5XX': {'description': 'Error with range, lower'}, '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}, 'default': {'description': 'A default response'}}, 'summary': 'C', 'operationId': 'c_c_get'}}, '/d': {'get': {'responses': {'400': {'description': 'Error with str'}, '5XX': {'description': 'Server Error', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ResponseModel'}}}}, '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}, 'default': {'description': 'Default Response', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ResponseModel'}}}}}, 'summary': 'D', 'operationId': 'd_d_get'}}}, 'components': {'schemas': {'ResponseModel': {'title': 'ResponseModel', 'required': ['message'], 'type': 'object', 'properties': {'message': {'title': 'Message', 'type': 'string'}}}}}}
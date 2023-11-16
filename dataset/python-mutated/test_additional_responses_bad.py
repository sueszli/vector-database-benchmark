import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI()

@app.get('/a', responses={'hello': {'description': 'Not a valid additional response'}})
async def a():
    pass
openapi_schema = {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/a': {'get': {'responses': {'hello': {'description': 'Not a valid additional response'}, '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'A', 'operationId': 'a_a_get'}}}}
client = TestClient(app)

def test_openapi_schema():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        client.get('/openapi.json')
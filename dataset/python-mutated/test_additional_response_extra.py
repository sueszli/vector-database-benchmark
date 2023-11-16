from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
router = APIRouter()
sub_router = APIRouter()
app = FastAPI()

@sub_router.get('/')
def read_item():
    if False:
        print('Hello World!')
    return {'id': 'foo'}
router.include_router(sub_router, prefix='/items')
app.include_router(router)
client = TestClient(app)

def test_path_operation():
    if False:
        while True:
            i = 10
    response = client.get('/items/')
    assert response.status_code == 200, response.text
    assert response.json() == {'id': 'foo'}

def test_openapi_schema():
    if False:
        i = 10
        return i + 15
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/items/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Read Item', 'operationId': 'read_item_items__get'}}}}
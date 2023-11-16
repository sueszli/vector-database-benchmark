import pytest
from fastapi import APIRouter, FastAPI
from fastapi.exceptions import FastAPIError
from fastapi.testclient import TestClient
app = FastAPI()
router = APIRouter()

@router.get('')
def get_empty():
    if False:
        i = 10
        return i + 15
    return ['OK']
app.include_router(router, prefix='/prefix')
client = TestClient(app)

def test_use_empty():
    if False:
        while True:
            i = 10
    with client:
        response = client.get('/prefix')
        assert response.status_code == 200, response.text
        assert response.json() == ['OK']
        response = client.get('/prefix/')
        assert response.status_code == 200, response.text
        assert response.json() == ['OK']

def test_include_empty():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FastAPIError):
        app.include_router(router)
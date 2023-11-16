from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

def test_redirect_slashes_enabled():
    if False:
        for i in range(10):
            print('nop')
    app = FastAPI()
    router = APIRouter()

    @router.get('/hello/')
    def hello_page() -> str:
        if False:
            while True:
                i = 10
        return 'Hello, World!'
    app.include_router(router)
    client = TestClient(app)
    response = client.get('/hello/', follow_redirects=False)
    assert response.status_code == 200
    response = client.get('/hello', follow_redirects=False)
    assert response.status_code == 307

def test_redirect_slashes_disabled():
    if False:
        i = 10
        return i + 15
    app = FastAPI(redirect_slashes=False)
    router = APIRouter()

    @router.get('/hello/')
    def hello_page() -> str:
        if False:
            print('Hello World!')
        return 'Hello, World!'
    app.include_router(router)
    client = TestClient(app)
    response = client.get('/hello/', follow_redirects=False)
    assert response.status_code == 200
    response = client.get('/hello', follow_redirects=False)
    assert response.status_code == 404
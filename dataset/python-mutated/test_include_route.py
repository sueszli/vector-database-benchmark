from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
app = FastAPI()
router = APIRouter()

@router.route('/items/')
def read_items(request: Request):
    if False:
        return 10
    return JSONResponse({'hello': 'world'})
app.include_router(router)
client = TestClient(app)

def test_sub_router():
    if False:
        while True:
            i = 10
    response = client.get('/items/')
    assert response.status_code == 200, response.text
    assert response.json() == {'hello': 'world'}
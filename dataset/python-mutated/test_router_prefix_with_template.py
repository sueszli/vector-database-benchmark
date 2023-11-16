from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
app = FastAPI()
router = APIRouter()

@router.get('/users/{id}')
def read_user(segment: str, id: str):
    if False:
        print('Hello World!')
    return {'segment': segment, 'id': id}
app.include_router(router, prefix='/{segment}')
client = TestClient(app)

def test_get():
    if False:
        i = 10
        return i + 15
    response = client.get('/seg/users/foo')
    assert response.status_code == 200, response.text
    assert response.json() == {'segment': 'seg', 'id': 'foo'}
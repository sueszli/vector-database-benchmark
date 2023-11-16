import pytest
from litestar import Litestar, MediaType, get
from litestar.status_codes import HTTP_200_OK
from litestar.testing import TestClient

@get(path='/health-check', media_type=MediaType.TEXT, sync_to_thread=False)
def health_check() -> str:
    if False:
        print('Hello World!')
    return 'healthy'
app = Litestar(route_handlers=[health_check])

def test_health_check() -> None:
    if False:
        i = 10
        return i + 15
    with TestClient(app=app) as client:
        response = client.get('/health-check')
        assert response.status_code == HTTP_200_OK
        assert response.text == 'healthy'

@pytest.fixture(scope='function')
def test_client() -> TestClient:
    if False:
        i = 10
        return i + 15
    return TestClient(app=app)

def test_health_check_with_fixture(test_client: TestClient) -> None:
    if False:
        for i in range(10):
            print('nop')
    with test_client as client:
        response = client.get('/health-check')
        assert response.status_code == HTTP_200_OK
        assert response.text == 'healthy'
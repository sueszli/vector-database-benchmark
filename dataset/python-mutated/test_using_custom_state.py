from docs.examples.application_state.using_custom_state import app
from litestar.status_codes import HTTP_200_OK
from litestar.testing import TestClient

def test_using_custom_state_example() -> None:
    if False:
        for i in range(10):
            print('nop')
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == HTTP_200_OK
        assert response.json() == {'count': 1}
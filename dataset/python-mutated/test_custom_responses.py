from docs.examples.responses.custom_responses import app as app_1
from litestar.testing import TestClient

def test_custom_responses() -> None:
    if False:
        for i in range(10):
            print('nop')
    with TestClient(app=app_1) as client:
        res = client.get('/')
        assert res.status_code == 200
        assert res.json() == {'foo': ['bar', 'baz']}
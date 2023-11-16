from dataclasses import asdict
from litestar import post
from litestar.params import Body
from litestar.status_codes import HTTP_201_CREATED
from litestar.testing import create_test_client
from . import Form

def test_request_body_json() -> None:
    if False:
        i = 10
        return i + 15

    @post(path='/test')
    def test_method(data: Form=Body()) -> None:
        if False:
            while True:
                i = 10
        assert isinstance(data, Form)
    with create_test_client(test_method) as client:
        response = client.post('/test', json=asdict(Form(name='Moishe Zuchmir', age=30, programmer=True, value='100')))
        assert response.status_code == HTTP_201_CREATED

def test_empty_dict_allowed() -> None:
    if False:
        for i in range(10):
            print('nop')

    @post(path='/test')
    def test_method(data: dict) -> None:
        if False:
            print('Hello World!')
        assert isinstance(data, dict)
    with create_test_client(test_method) as client:
        response = client.post('/test', json={})
        assert response.status_code == HTTP_201_CREATED
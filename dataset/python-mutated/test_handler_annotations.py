from uuid import UUID
import pytest
from sanic import json

@pytest.mark.parametrize('idx,path,expectation', ((0, '/abc', 'str'), (1, '/123', 'int'), (2, '/123.5', 'float'), (3, '/8af729fe-2b94-4a95-a168-c07068568429', 'UUID')))
def test_annotated_handlers(app, idx, path, expectation):
    if False:
        print('Hello World!')

    def build_response(num, foo):
        if False:
            return 10
        return json({'num': num, 'type': type(foo).__name__})

    @app.get('/<foo>')
    def handler0(_, foo: str):
        if False:
            return 10
        return build_response(0, foo)

    @app.get('/<foo>')
    def handler1(_, foo: int):
        if False:
            print('Hello World!')
        return build_response(1, foo)

    @app.get('/<foo>')
    def handler2(_, foo: float):
        if False:
            return 10
        return build_response(2, foo)

    @app.get('/<foo>')
    def handler3(_, foo: UUID):
        if False:
            return 10
        return build_response(3, foo)
    (_, response) = app.test_client.get(path)
    assert response.json['num'] == idx
    assert response.json['type'] == expectation
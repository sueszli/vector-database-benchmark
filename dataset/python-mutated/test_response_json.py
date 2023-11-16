import json
from functools import partial
from unittest.mock import Mock
import pytest
from sanic import Request, Sanic
from sanic.exceptions import SanicException
from sanic.response import json as json_response
from sanic.response.types import JSONResponse
JSON_BODY = {'ok': True}
json_dumps = partial(json.dumps, separators=(',', ':'))

@pytest.fixture
def json_app(app: Sanic):
    if False:
        while True:
            i = 10

    @app.get('/json')
    async def handle(request: Request):
        return json_response(JSON_BODY)
    return app

def test_body_can_be_retrieved(json_app: Sanic):
    if False:
        print('Hello World!')
    (_, resp) = json_app.test_client.get('/json')
    assert resp.body == json_dumps(JSON_BODY).encode()

def test_body_can_be_set(json_app: Sanic):
    if False:
        print('Hello World!')
    new_body = b'{"hello":"world"}'

    @json_app.on_response
    def set_body(request: Request, response: JSONResponse):
        if False:
            while True:
                i = 10
        response.body = new_body
    (_, resp) = json_app.test_client.get('/json')
    assert resp.body == new_body

def test_raw_body_can_be_retrieved(json_app: Sanic):
    if False:
        print('Hello World!')

    @json_app.on_response
    def check_body(request: Request, response: JSONResponse):
        if False:
            return 10
        assert response.raw_body == JSON_BODY
    json_app.test_client.get('/json')

def test_raw_body_can_be_set(json_app: Sanic):
    if False:
        while True:
            i = 10
    new_body = {'hello': 'world'}

    @json_app.on_response
    def set_body(request: Request, response: JSONResponse):
        if False:
            while True:
                i = 10
        response.raw_body = new_body
        assert response.raw_body == new_body
        assert response.body == json_dumps(new_body).encode()
    json_app.test_client.get('/json')

def test_raw_body_cant_be_retrieved_after_body_set(json_app: Sanic):
    if False:
        i = 10
        return i + 15
    new_body = b'{"hello":"world"}'

    @json_app.on_response
    def check_raw_body(request: Request, response: JSONResponse):
        if False:
            print('Hello World!')
        response.body = new_body
        with pytest.raises(SanicException):
            response.raw_body
    json_app.test_client.get('/json')

def test_raw_body_can_be_reset_after_body_set(json_app: Sanic):
    if False:
        return 10
    new_body = b'{"hello":"world"}'
    new_new_body = {'lorem': 'ipsum'}

    @json_app.on_response
    def set_bodies(request: Request, response: JSONResponse):
        if False:
            print('Hello World!')
        response.body = new_body
        response.raw_body = new_new_body
    (_, resp) = json_app.test_client.get('/json')
    assert resp.body == json_dumps(new_new_body).encode()

def test_set_body_method(json_app: Sanic):
    if False:
        for i in range(10):
            print('nop')
    new_body = {'lorem': 'ipsum'}

    @json_app.on_response
    def set_body(request: Request, response: JSONResponse):
        if False:
            while True:
                i = 10
        response.set_body(new_body)
    (_, resp) = json_app.test_client.get('/json')
    assert resp.body == json_dumps(new_body).encode()

def test_set_body_method_after_body_set(json_app: Sanic):
    if False:
        while True:
            i = 10
    new_body = b'{"hello":"world"}'
    new_new_body = {'lorem': 'ipsum'}

    @json_app.on_response
    def set_body(request: Request, response: JSONResponse):
        if False:
            i = 10
            return i + 15
        response.body = new_body
        response.set_body(new_new_body)
    (_, resp) = json_app.test_client.get('/json')
    assert resp.body == json_dumps(new_new_body).encode()

def test_custom_dumps_and_kwargs(json_app: Sanic):
    if False:
        print('Hello World!')
    custom_dumps = Mock(return_value='custom')

    @json_app.get('/json-custom')
    async def handle_custom(request: Request):
        return json_response(JSON_BODY, dumps=custom_dumps, prry='platypus')
    (_, resp) = json_app.test_client.get('/json-custom')
    assert resp.body == 'custom'.encode()
    custom_dumps.assert_called_once_with(JSON_BODY, prry='platypus')

def test_override_dumps_and_kwargs(json_app: Sanic):
    if False:
        i = 10
        return i + 15
    custom_dumps_1 = Mock(return_value='custom1')
    custom_dumps_2 = Mock(return_value='custom2')

    @json_app.get('/json-custom')
    async def handle_custom(request: Request):
        return json_response(JSON_BODY, dumps=custom_dumps_1, prry='platypus')

    @json_app.on_response
    def set_body(request: Request, response: JSONResponse):
        if False:
            while True:
                i = 10
        response.set_body(JSON_BODY, dumps=custom_dumps_2, platypus='prry')
    (_, resp) = json_app.test_client.get('/json-custom')
    assert resp.body == 'custom2'.encode()
    custom_dumps_1.assert_called_once_with(JSON_BODY, prry='platypus')
    custom_dumps_2.assert_called_once_with(JSON_BODY, platypus='prry')

def test_append(json_app: Sanic):
    if False:
        return 10

    @json_app.get('/json-append')
    async def handler_append(request: Request):
        return json_response(['a', 'b'], status=200)

    @json_app.on_response
    def do_append(request: Request, response: JSONResponse):
        if False:
            for i in range(10):
                print('nop')
        response.append('c')
    (_, resp) = json_app.test_client.get('/json-append')
    assert resp.body == json_dumps(['a', 'b', 'c']).encode()

def test_extend(json_app: Sanic):
    if False:
        for i in range(10):
            print('nop')

    @json_app.get('/json-extend')
    async def handler_extend(request: Request):
        return json_response(['a', 'b'], status=200)

    @json_app.on_response
    def do_extend(request: Request, response: JSONResponse):
        if False:
            for i in range(10):
                print('nop')
        response.extend(['c', 'd'])
    (_, resp) = json_app.test_client.get('/json-extend')
    assert resp.body == json_dumps(['a', 'b', 'c', 'd']).encode()

def test_update(json_app: Sanic):
    if False:
        for i in range(10):
            print('nop')

    @json_app.get('/json-update')
    async def handler_update(request: Request):
        return json_response({'a': 'b'}, status=200)

    @json_app.on_response
    def do_update(request: Request, response: JSONResponse):
        if False:
            print('Hello World!')
        response.update({'c': 'd'}, e='f')
    (_, resp) = json_app.test_client.get('/json-update')
    assert resp.body == json_dumps({'a': 'b', 'c': 'd', 'e': 'f'}).encode()

def test_pop_dict(json_app: Sanic):
    if False:
        for i in range(10):
            print('nop')

    @json_app.get('/json-pop')
    async def handler_pop(request: Request):
        return json_response({'a': 'b', 'c': 'd'}, status=200)

    @json_app.on_response
    def do_pop(request: Request, response: JSONResponse):
        if False:
            while True:
                i = 10
        val = response.pop('c')
        assert val == 'd'
        val_default = response.pop('e', 'f')
        assert val_default == 'f'
    (_, resp) = json_app.test_client.get('/json-pop')
    assert resp.body == json_dumps({'a': 'b'}).encode()

def test_pop_list(json_app: Sanic):
    if False:
        while True:
            i = 10

    @json_app.get('/json-pop')
    async def handler_pop(request: Request):
        return json_response(['a', 'b'], status=200)

    @json_app.on_response
    def do_pop(request: Request, response: JSONResponse):
        if False:
            print('Hello World!')
        val = response.pop(0)
        assert val == 'a'
        with pytest.raises(TypeError, match="pop doesn't accept a default argument for lists"):
            response.pop(21, 'nah nah')
    (_, resp) = json_app.test_client.get('/json-pop')
    assert resp.body == json_dumps(['b']).encode()

def test_json_response_class_sets_proper_content_type(json_app: Sanic):
    if False:
        for i in range(10):
            print('nop')

    @json_app.get('/json-class')
    async def handler(request: Request):
        return JSONResponse(JSON_BODY)
    (_, resp) = json_app.test_client.get('/json-class')
    assert resp.headers['content-type'] == 'application/json'
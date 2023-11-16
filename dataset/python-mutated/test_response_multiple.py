from typing import List, Union
import pytest
from pydantic import ValidationError
from ninja import NinjaAPI, Schema
from ninja.errors import ConfigError
from ninja.responses import codes_2xx, codes_3xx
from ninja.testing import TestClient
api = NinjaAPI()

@api.get('/check_int', response={200: int})
def check_int(request):
    if False:
        return 10
    return (200, '1')

@api.get('/check_int2', response={200: int})
def check_int2(request):
    if False:
        for i in range(10):
            print('nop')
    return (200, 'str')

@api.get('/check_single_with_status', response=int)
def check_single_with_status(request, code: int):
    if False:
        return 10
    return (code, 1)

@api.get('/check_response_schema', response={400: int})
def check_response_schema(request):
    if False:
        print('Hello World!')
    return (200, 1)

@api.get('/check_no_content', response={204: None})
def check_no_content(request, return_code: bool):
    if False:
        while True:
            i = 10
    if return_code:
        return (204, None)
    return

@api.get('/check_multiple_codes', response={codes_2xx: int, codes_3xx: str, ...: float})
def check_multiple_codes(request, code: int):
    if False:
        while True:
            i = 10
    return (code, '1')

class User:

    def __init__(self, id, name, password):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        self.name = name
        self.password = password

class UserModel(Schema):
    id: int
    name: str

class ErrorModel(Schema):
    detail: str

@api.get('/check_model', response={200: UserModel, 202: UserModel})
def check_model(request):
    if False:
        print('Hello World!')
    return (202, User(1, 'John', 'Password'))

@api.get('/check_list_model', response={200: List[UserModel]})
def check_list_model(request):
    if False:
        for i in range(10):
            print('nop')
    return (200, [User(1, 'John', 'Password')])

@api.get('/check_union', response={200: Union[int, UserModel], 400: ErrorModel})
def check_union(request, q: int):
    if False:
        return 10
    if q == 0:
        return (200, 1)
    if q == 1:
        return (200, User(1, 'John', 'Password'))
    if q == 2:
        return (400, {'detail': 'error'})
    return 'invalid'
client = TestClient(api)

@pytest.mark.parametrize('path,expected_status,expected_response', [('/check_int', 200, 1), ('/check_single_with_status?code=200', 200, 1), ('/check_model', 202, {'id': 1, 'name': 'John'}), ('/check_list_model', 200, [{'id': 1, 'name': 'John'}]), ('/check_union?q=0', 200, 1), ('/check_union?q=1', 200, {'id': 1, 'name': 'John'}), ('/check_union?q=2', 400, {'detail': 'error'}), ('/check_multiple_codes?code=200', 200, 1), ('/check_multiple_codes?code=201', 201, 1), ('/check_multiple_codes?code=202', 202, 1), ('/check_multiple_codes?code=206', 206, 1), ('/check_multiple_codes?code=300', 300, '1'), ('/check_multiple_codes?code=308', 308, '1'), ('/check_multiple_codes?code=400', 400, 1.0), ('/check_multiple_codes?code=500', 500, 1.0)])
def test_responses(path, expected_status, expected_response):
    if False:
        for i in range(10):
            print('nop')
    response = client.get(path)
    assert response.status_code == expected_status, response.content
    assert response.json() == expected_response

def test_schema():
    if False:
        for i in range(10):
            print('nop')
    checks = [('/api/check_int', {200}), ('/api/check_int2', {200}), ('/api/check_single_with_status', {200}), ('/api/check_response_schema', {400}), ('/api/check_model', {200, 202}), ('/api/check_list_model', {200}), ('/api/check_union', {200, 400})]
    schema = api.get_openapi_schema()
    for (path, codes) in checks:
        responses = schema['paths'][path]['get']['responses']
        responses_codes = set(responses.keys())
        assert codes == responses_codes, f'{codes} != {responses_codes}'
    check_model_responses = schema['paths']['/api/check_model']['get']['responses']
    assert check_model_responses == {200: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/UserModel'}}}, 'description': 'OK'}, 202: {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/UserModel'}}}, 'description': 'Accepted'}}

def test_no_content():
    if False:
        i = 10
        return i + 15
    response = client.get('/check_no_content?return_code=1')
    assert response.status_code == 204
    assert response.content == b''
    response = client.get('/check_no_content?return_code=0')
    assert response.status_code == 204
    assert response.content == b''
    schema = api.get_openapi_schema()
    details = schema['paths']['/api/check_no_content']['get']['responses']
    assert details == {204: {'description': 'No Content'}}

def test_validates():
    if False:
        while True:
            i = 10
    with pytest.raises(ValidationError):
        client.get('/check_int2')
    with pytest.raises(ValidationError):
        client.get('/check_union?q=3')
    with pytest.raises(ConfigError):
        client.get('/check_response_schema')
    with pytest.raises(ConfigError):
        client.get('/check_single_with_status?code=300')
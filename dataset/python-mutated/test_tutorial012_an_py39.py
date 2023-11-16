import pytest
from dirty_equals import IsDict
from fastapi.testclient import TestClient
from fastapi.utils import match_pydantic_error_url
from ...utils import needs_py39

@pytest.fixture(name='client')
def get_client():
    if False:
        i = 10
        return i + 15
    from docs_src.dependencies.tutorial012_an_py39 import app
    client = TestClient(app)
    return client

@needs_py39
def test_get_no_headers_items(client: TestClient):
    if False:
        i = 10
        return i + 15
    response = client.get('/items/')
    assert response.status_code == 422, response.text
    assert response.json() == IsDict({'detail': [{'type': 'missing', 'loc': ['header', 'x-token'], 'msg': 'Field required', 'input': None, 'url': match_pydantic_error_url('missing')}, {'type': 'missing', 'loc': ['header', 'x-key'], 'msg': 'Field required', 'input': None, 'url': match_pydantic_error_url('missing')}]}) | IsDict({'detail': [{'loc': ['header', 'x-token'], 'msg': 'field required', 'type': 'value_error.missing'}, {'loc': ['header', 'x-key'], 'msg': 'field required', 'type': 'value_error.missing'}]})

@needs_py39
def test_get_no_headers_users(client: TestClient):
    if False:
        i = 10
        return i + 15
    response = client.get('/users/')
    assert response.status_code == 422, response.text
    assert response.json() == IsDict({'detail': [{'type': 'missing', 'loc': ['header', 'x-token'], 'msg': 'Field required', 'input': None, 'url': match_pydantic_error_url('missing')}, {'type': 'missing', 'loc': ['header', 'x-key'], 'msg': 'Field required', 'input': None, 'url': match_pydantic_error_url('missing')}]}) | IsDict({'detail': [{'loc': ['header', 'x-token'], 'msg': 'field required', 'type': 'value_error.missing'}, {'loc': ['header', 'x-key'], 'msg': 'field required', 'type': 'value_error.missing'}]})

@needs_py39
def test_get_invalid_one_header_items(client: TestClient):
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/items/', headers={'X-Token': 'invalid'})
    assert response.status_code == 400, response.text
    assert response.json() == {'detail': 'X-Token header invalid'}

@needs_py39
def test_get_invalid_one_users(client: TestClient):
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/users/', headers={'X-Token': 'invalid'})
    assert response.status_code == 400, response.text
    assert response.json() == {'detail': 'X-Token header invalid'}

@needs_py39
def test_get_invalid_second_header_items(client: TestClient):
    if False:
        i = 10
        return i + 15
    response = client.get('/items/', headers={'X-Token': 'fake-super-secret-token', 'X-Key': 'invalid'})
    assert response.status_code == 400, response.text
    assert response.json() == {'detail': 'X-Key header invalid'}

@needs_py39
def test_get_invalid_second_header_users(client: TestClient):
    if False:
        i = 10
        return i + 15
    response = client.get('/users/', headers={'X-Token': 'fake-super-secret-token', 'X-Key': 'invalid'})
    assert response.status_code == 400, response.text
    assert response.json() == {'detail': 'X-Key header invalid'}

@needs_py39
def test_get_valid_headers_items(client: TestClient):
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/items/', headers={'X-Token': 'fake-super-secret-token', 'X-Key': 'fake-super-secret-key'})
    assert response.status_code == 200, response.text
    assert response.json() == [{'item': 'Portal Gun'}, {'item': 'Plumbus'}]

@needs_py39
def test_get_valid_headers_users(client: TestClient):
    if False:
        return 10
    response = client.get('/users/', headers={'X-Token': 'fake-super-secret-token', 'X-Key': 'fake-super-secret-key'})
    assert response.status_code == 200, response.text
    assert response.json() == [{'username': 'Rick'}, {'username': 'Morty'}]

@needs_py39
def test_openapi_schema(client: TestClient):
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/items/': {'get': {'summary': 'Read Items', 'operationId': 'read_items_items__get', 'parameters': [{'required': True, 'schema': {'title': 'X-Token', 'type': 'string'}, 'name': 'x-token', 'in': 'header'}, {'required': True, 'schema': {'title': 'X-Key', 'type': 'string'}, 'name': 'x-key', 'in': 'header'}], 'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}, '422': {'description': 'Validation Error', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/HTTPValidationError'}}}}}}}, '/users/': {'get': {'summary': 'Read Users', 'operationId': 'read_users_users__get', 'parameters': [{'required': True, 'schema': {'title': 'X-Token', 'type': 'string'}, 'name': 'x-token', 'in': 'header'}, {'required': True, 'schema': {'title': 'X-Key', 'type': 'string'}, 'name': 'x-key', 'in': 'header'}], 'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}, '422': {'description': 'Validation Error', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/HTTPValidationError'}}}}}}}}, 'components': {'schemas': {'HTTPValidationError': {'title': 'HTTPValidationError', 'type': 'object', 'properties': {'detail': {'title': 'Detail', 'type': 'array', 'items': {'$ref': '#/components/schemas/ValidationError'}}}}, 'ValidationError': {'title': 'ValidationError', 'required': ['loc', 'msg', 'type'], 'type': 'object', 'properties': {'loc': {'title': 'Location', 'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}, 'msg': {'title': 'Message', 'type': 'string'}, 'type': {'title': 'Error Type', 'type': 'string'}}}}}}
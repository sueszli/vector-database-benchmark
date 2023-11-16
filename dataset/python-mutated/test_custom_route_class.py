import pytest
from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Route
app = FastAPI()

class APIRouteA(APIRoute):
    x_type = 'A'

class APIRouteB(APIRoute):
    x_type = 'B'

class APIRouteC(APIRoute):
    x_type = 'C'
router_a = APIRouter(route_class=APIRouteA)
router_b = APIRouter(route_class=APIRouteB)
router_c = APIRouter(route_class=APIRouteC)

@router_a.get('/')
def get_a():
    if False:
        for i in range(10):
            print('nop')
    return {'msg': 'A'}

@router_b.get('/')
def get_b():
    if False:
        print('Hello World!')
    return {'msg': 'B'}

@router_c.get('/')
def get_c():
    if False:
        while True:
            i = 10
    return {'msg': 'C'}
router_b.include_router(router=router_c, prefix='/c')
router_a.include_router(router=router_b, prefix='/b')
app.include_router(router=router_a, prefix='/a')
client = TestClient(app)

@pytest.mark.parametrize('path,expected_status,expected_response', [('/a', 200, {'msg': 'A'}), ('/a/b', 200, {'msg': 'B'}), ('/a/b/c', 200, {'msg': 'C'})])
def test_get_path(path, expected_status, expected_response):
    if False:
        return 10
    response = client.get(path)
    assert response.status_code == expected_status
    assert response.json() == expected_response

def test_route_classes():
    if False:
        print('Hello World!')
    routes = {}
    for r in app.router.routes:
        assert isinstance(r, Route)
        routes[r.path] = r
    assert getattr(routes['/a/'], 'x_type') == 'A'
    assert getattr(routes['/a/b/'], 'x_type') == 'B'
    assert getattr(routes['/a/b/c/'], 'x_type') == 'C'

def test_openapi_schema():
    if False:
        i = 10
        return i + 15
    response = client.get('/openapi.json')
    assert response.status_code == 200, response.text
    assert response.json() == {'openapi': '3.1.0', 'info': {'title': 'FastAPI', 'version': '0.1.0'}, 'paths': {'/a/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Get A', 'operationId': 'get_a_a__get'}}, '/a/b/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Get B', 'operationId': 'get_b_a_b__get'}}, '/a/b/c/': {'get': {'responses': {'200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}}}, 'summary': 'Get C', 'operationId': 'get_c_a_b_c__get'}}}}
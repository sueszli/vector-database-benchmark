import pytest
from ninja import NinjaAPI, Router
from ninja.testing import TestClient
api = NinjaAPI()

@api.get('/endpoint')
def global_op(request):
    if False:
        print('Hello World!')
    return 'global'
first_router = Router()

@first_router.get('/endpoint_1')
def router_op1(request):
    if False:
        i = 10
        return i + 15
    return 'first 1'
second_router_one = Router()

@second_router_one.get('endpoint_1')
def router_op2(request):
    if False:
        print('Hello World!')
    return 'second 1'
second_router_two = Router()

@second_router_two.get('endpoint_2')
def router2_op3(request):
    if False:
        for i in range(10):
            print('nop')
    return 'second 2'
first_router.add_router('/second', second_router_one, tags=['one'])
first_router.add_router('/second', second_router_two, tags=['two'])
api.add_router('/first', first_router, tags=['global'])

@first_router.get('endpoint_2')
def router1_op1(request):
    if False:
        print('Hello World!')
    return 'first 2'

@second_router_one.get('endpoint_3')
def router21_op3(request, path_param: int=None):
    if False:
        i = 10
        return i + 15
    return 'second 3' if path_param is None else f'second 3: {path_param}'
second_router_three = Router()

@second_router_three.get('endpoint_4')
def router_op3(request, path_param: int=None):
    if False:
        i = 10
        return i + 15
    return 'second 4' if path_param is None else f'second 4: {path_param}'
first_router.add_router('/second', second_router_three, tags=['three'])
client = TestClient(api)

@pytest.mark.parametrize('path,expected_status,expected_response', [('/endpoint', 200, 'global'), ('/first/endpoint_1', 200, 'first 1'), ('/first/endpoint_2', 200, 'first 2'), ('/first/second/endpoint_1', 200, 'second 1'), ('/first/second/endpoint_2', 200, 'second 2'), ('/first/second/endpoint_3', 200, 'second 3'), ('/first/second/endpoint_4', 200, 'second 4')])
def test_inheritance_responses(path, expected_status, expected_response):
    if False:
        return 10
    response = client.get(path)
    assert response.status_code == expected_status, response.content
    assert response.json() == expected_response

def test_tags():
    if False:
        return 10
    schema = api.get_openapi_schema()
    glob = schema['paths']['/api/first/endpoint_1']['get']
    assert glob['tags'] == ['global']
    e1 = schema['paths']['/api/first/second/endpoint_1']['get']
    assert e1['tags'] == ['one']
    e2 = schema['paths']['/api/first/second/endpoint_2']['get']
    assert e2['tags'] == ['two']
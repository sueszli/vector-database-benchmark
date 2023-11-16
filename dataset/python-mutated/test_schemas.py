from starlette.applications import Starlette
from starlette.endpoints import HTTPEndpoint
from starlette.routing import Host, Mount, Route, Router, WebSocketRoute
from starlette.schemas import SchemaGenerator
schemas = SchemaGenerator({'openapi': '3.0.0', 'info': {'title': 'Example API', 'version': '1.0'}})

def ws(session):
    if False:
        print('Hello World!')
    'ws'
    pass

def get_user(request):
    if False:
        i = 10
        return i + 15
    '\n    responses:\n        200:\n            description: A user.\n            examples:\n                {"username": "tom"}\n    '
    pass

def list_users(request):
    if False:
        while True:
            i = 10
    '\n    responses:\n      200:\n        description: A list of users.\n        examples:\n          [{"username": "tom"}, {"username": "lucy"}]\n    '
    pass

def create_user(request):
    if False:
        while True:
            i = 10
    '\n    responses:\n      200:\n        description: A user.\n        examples:\n          {"username": "tom"}\n    '
    pass

class OrganisationsEndpoint(HTTPEndpoint):

    def get(self, request):
        if False:
            print('Hello World!')
        '\n        responses:\n          200:\n            description: A list of organisations.\n            examples:\n              [{"name": "Foo Corp."}, {"name": "Acme Ltd."}]\n        '
        pass

    def post(self, request):
        if False:
            while True:
                i = 10
        '\n        responses:\n          200:\n            description: An organisation.\n            examples:\n              {"name": "Foo Corp."}\n        '
        pass

def regular_docstring_and_schema(request):
    if False:
        i = 10
        return i + 15
    '\n    This a regular docstring example (not included in schema)\n\n    ---\n\n    responses:\n      200:\n        description: This is included in the schema.\n    '
    pass

def regular_docstring(request):
    if False:
        return 10
    '\n    This a regular docstring example (not included in schema)\n    '
    pass

def no_docstring(request):
    if False:
        return 10
    pass

def subapp_endpoint(request):
    if False:
        print('Hello World!')
    '\n    responses:\n      200:\n        description: This endpoint is part of a subapp.\n    '
    pass

def schema(request):
    if False:
        return 10
    return schemas.OpenAPIResponse(request=request)
subapp = Starlette(routes=[Route('/subapp-endpoint', endpoint=subapp_endpoint)])
app = Starlette(routes=[WebSocketRoute('/ws', endpoint=ws), Route('/users/{id:int}', endpoint=get_user, methods=['GET']), Route('/users', endpoint=list_users, methods=['GET', 'HEAD']), Route('/users', endpoint=create_user, methods=['POST']), Route('/orgs', endpoint=OrganisationsEndpoint), Route('/regular-docstring-and-schema', endpoint=regular_docstring_and_schema), Route('/regular-docstring', endpoint=regular_docstring), Route('/no-docstring', endpoint=no_docstring), Route('/schema', endpoint=schema, methods=['GET'], include_in_schema=False), Mount('/subapp', subapp), Host('sub.domain.com', app=Router(routes=[Mount('/subapp2', subapp)]))])

def test_schema_generation():
    if False:
        print('Hello World!')
    schema = schemas.get_schema(routes=app.routes)
    assert schema == {'openapi': '3.0.0', 'info': {'title': 'Example API', 'version': '1.0'}, 'paths': {'/orgs': {'get': {'responses': {200: {'description': 'A list of organisations.', 'examples': [{'name': 'Foo Corp.'}, {'name': 'Acme Ltd.'}]}}}, 'post': {'responses': {200: {'description': 'An organisation.', 'examples': {'name': 'Foo Corp.'}}}}}, '/regular-docstring-and-schema': {'get': {'responses': {200: {'description': 'This is included in the schema.'}}}}, '/subapp/subapp-endpoint': {'get': {'responses': {200: {'description': 'This endpoint is part of a subapp.'}}}}, '/subapp2/subapp-endpoint': {'get': {'responses': {200: {'description': 'This endpoint is part of a subapp.'}}}}, '/users': {'get': {'responses': {200: {'description': 'A list of users.', 'examples': [{'username': 'tom'}, {'username': 'lucy'}]}}}, 'post': {'responses': {200: {'description': 'A user.', 'examples': {'username': 'tom'}}}}}, '/users/{id}': {'get': {'responses': {200: {'description': 'A user.', 'examples': {'username': 'tom'}}}}}}}
EXPECTED_SCHEMA = "\ninfo:\n  title: Example API\n  version: '1.0'\nopenapi: 3.0.0\npaths:\n  /orgs:\n    get:\n      responses:\n        200:\n          description: A list of organisations.\n          examples:\n          - name: Foo Corp.\n          - name: Acme Ltd.\n    post:\n      responses:\n        200:\n          description: An organisation.\n          examples:\n            name: Foo Corp.\n  /regular-docstring-and-schema:\n    get:\n      responses:\n        200:\n          description: This is included in the schema.\n  /subapp/subapp-endpoint:\n    get:\n      responses:\n        200:\n          description: This endpoint is part of a subapp.\n  /subapp2/subapp-endpoint:\n    get:\n      responses:\n        200:\n          description: This endpoint is part of a subapp.\n  /users:\n    get:\n      responses:\n        200:\n          description: A list of users.\n          examples:\n          - username: tom\n          - username: lucy\n    post:\n      responses:\n        200:\n          description: A user.\n          examples:\n            username: tom\n  /users/{id}:\n    get:\n      responses:\n        200:\n          description: A user.\n          examples:\n            username: tom\n"

def test_schema_endpoint(test_client_factory):
    if False:
        return 10
    client = test_client_factory(app)
    response = client.get('/schema')
    assert response.headers['Content-Type'] == 'application/vnd.oai.openapi'
    assert response.text.strip() == EXPECTED_SCHEMA.strip()
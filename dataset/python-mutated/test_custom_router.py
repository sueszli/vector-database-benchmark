import pytest
import falcon
from falcon import testing
from _util import create_app

@pytest.mark.parametrize('asgi', [True, False])
def test_custom_router_add_route_should_be_used(asgi):
    if False:
        print('Hello World!')
    check = []

    class CustomRouter:

        def add_route(self, uri_template, *args, **kwargs):
            if False:
                return 10
            check.append(uri_template)

        def find(self, uri):
            if False:
                print('Hello World!')
            pass
    app = create_app(asgi=asgi, router=CustomRouter())
    app.add_route('/test', 'resource')
    assert len(check) == 1
    assert '/test' in check

@pytest.mark.parametrize('asgi', [True, False])
def test_custom_router_find_should_be_used(asgi):
    if False:
        while True:
            i = 10
    if asgi:

        async def resource(req, resp, **kwargs):
            resp.text = '{{"uri_template": "{0}"}}'.format(req.uri_template)
    else:

        def resource(req, resp, **kwargs):
            if False:
                while True:
                    i = 10
            resp.text = '{{"uri_template": "{0}"}}'.format(req.uri_template)

    class CustomRouter:

        def __init__(self):
            if False:
                print('Hello World!')
            self.reached_backwards_compat = False

        def find(self, uri, req=None):
            if False:
                while True:
                    i = 10
            if uri == '/test/42':
                return (resource, {'GET': resource}, {}, '/test/{id}')
            if uri == '/test/42/no-uri-template':
                return (resource, {'GET': resource}, {}, None)
            if uri == '/test/42/uri-template/backwards-compat':
                return (resource, {'GET': resource}, {})
            if uri == '/404/backwards-compat':
                self.reached_backwards_compat = True
                return (None, None, None)
            return None
    router = CustomRouter()
    app = create_app(asgi=asgi, router=router)
    client = testing.TestClient(app)
    response = client.simulate_request(path='/test/42')
    assert response.content == b'{"uri_template": "/test/{id}"}'
    response = client.simulate_request(path='/test/42/no-uri-template')
    assert response.content == b'{"uri_template": "None"}'
    response = client.simulate_request(path='/test/42/uri-template/backwards-compat')
    assert response.content == b'{"uri_template": "None"}'
    for uri in ('/404', '/404/backwards-compat'):
        response = client.simulate_request(path=uri)
        assert response.content == falcon.HTTPNotFound().to_json()
        assert response.status == falcon.HTTP_404
    assert router.reached_backwards_compat

@pytest.mark.parametrize('asgi', [True, False])
def test_can_pass_additional_params_to_add_route(asgi):
    if False:
        while True:
            i = 10
    check = []

    class CustomRouter:

        def add_route(self, uri_template, resource, **kwargs):
            if False:
                return 10
            name = kwargs['name']
            self._index = {name: uri_template}
            check.append(name)

        def find(self, uri):
            if False:
                while True:
                    i = 10
            pass
    app = create_app(asgi=asgi, router=CustomRouter())
    app.add_route('/test', 'resource', name='my-url-name')
    assert len(check) == 1
    assert 'my-url-name' in check
    with pytest.raises(TypeError):
        app.add_route('/test', 'resource', 'xarg1', 'xarg2')

@pytest.mark.parametrize('asgi', [True, False])
def test_custom_router_takes_req_positional_argument(asgi):
    if False:
        print('Hello World!')
    if asgi:

        async def responder(req, resp):
            resp.text = 'OK'
    else:

        def responder(req, resp):
            if False:
                while True:
                    i = 10
            resp.text = 'OK'

    class CustomRouter:

        def find(self, uri, req):
            if False:
                i = 10
                return i + 15
            if uri == '/test' and isinstance(req, falcon.Request):
                return (responder, {'GET': responder}, {}, None)
    router = CustomRouter()
    app = create_app(asgi=asgi, router=router)
    client = testing.TestClient(app)
    response = client.simulate_request(path='/test')
    assert response.content == b'OK'

@pytest.mark.parametrize('asgi', [True, False])
def test_custom_router_takes_req_keyword_argument(asgi):
    if False:
        for i in range(10):
            print('nop')
    if asgi:

        async def responder(req, resp):
            resp.text = 'OK'
    else:

        def responder(req, resp):
            if False:
                while True:
                    i = 10
            resp.text = 'OK'

    class CustomRouter:

        def find(self, uri, req=None):
            if False:
                while True:
                    i = 10
            if uri == '/test' and isinstance(req, falcon.Request):
                return (responder, {'GET': responder}, {}, None)
    router = CustomRouter()
    app = create_app(asgi=asgi, router=router)
    client = testing.TestClient(app)
    response = client.simulate_request(path='/test')
    assert response.content == b'OK'
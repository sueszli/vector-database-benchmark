import importlib
import os
import wsgiref.validate
import pytest
import falcon
from falcon import testing
import falcon.constants
from falcon.routing import util
from _util import create_app, has_cython
FALCON_CUSTOM_HTTP_METHODS = ['FOO', 'BAR']

@pytest.fixture
def resource_things():
    if False:
        print('Hello World!')
    return ThingsResource()

@pytest.fixture
def cleanup_constants():
    if False:
        print('Hello World!')
    importlib.reload(falcon.constants)
    orig = list(falcon.constants.COMBINED_METHODS)
    yield
    falcon.constants.COMBINED_METHODS = orig
    if 'FALCON_CUSTOM_HTTP_METHODS' in os.environ:
        del os.environ['FALCON_CUSTOM_HTTP_METHODS']

@pytest.fixture
def custom_http_client(asgi, request, cleanup_constants, resource_things):
    if False:
        i = 10
        return i + 15
    falcon.constants.COMBINED_METHODS += FALCON_CUSTOM_HTTP_METHODS
    app = create_app(asgi)
    app.add_route('/things', resource_things)
    return testing.TestClient(app)

class ThingsResource:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.called = False
        self.on_patch = {}

    def on_foo(self, req, resp):
        if False:
            while True:
                i = 10
        self.called = True
        (self.req, self.resp) = (req, resp)
        resp.status = falcon.HTTP_204

def test_map_http_methods(custom_http_client, resource_things):
    if False:
        for i in range(10):
            print('nop')
    method_map = util.map_http_methods(resource_things)
    assert 'FOO' in method_map
    assert 'BAR' not in method_map

@pytest.mark.skipif(has_cython, reason='Reloading modules on Cython does not work')
@pytest.mark.parametrize('env_str,expected', [('foo', ['FOO']), ('FOO', ['FOO']), ('FOO,', ['FOO']), ('FOO,BAR', ['FOO', 'BAR']), ('FOO, BAR', ['FOO', 'BAR']), (' foo , BAR ', ['FOO', 'BAR'])])
def test_environment_override(cleanup_constants, resource_things, env_str, expected):
    if False:
        while True:
            i = 10
    for method in expected:
        assert method not in falcon.constants.COMBINED_METHODS
    os.environ['FALCON_CUSTOM_HTTP_METHODS'] = env_str
    importlib.reload(falcon.constants)
    for method in expected:
        assert method in falcon.constants.COMBINED_METHODS

def test_foo(custom_http_client, resource_things):
    if False:
        i = 10
        return i + 15
    'FOO is a supported method, so returns HTTP_204'
    custom_http_client.app.add_route('/things', resource_things)

    def s():
        if False:
            while True:
                i = 10
        return custom_http_client.simulate_request(path='/things', method='FOO')
    if not custom_http_client.app._ASGI:
        with pytest.warns(wsgiref.validate.WSGIWarning):
            response = s()
    else:
        response = s()
    assert 'FOO' in falcon.constants.COMBINED_METHODS
    assert response.status == falcon.HTTP_204
    assert response.status_code == 204
    assert resource_things.called

def test_bar(custom_http_client, resource_things):
    if False:
        while True:
            i = 10
    'BAR is not supported by ResourceThing'
    custom_http_client.app.add_route('/things', resource_things)

    def s():
        if False:
            for i in range(10):
                print('nop')
        return custom_http_client.simulate_request(path='/things', method='BAR')
    if not custom_http_client.app._ASGI:
        with pytest.warns(wsgiref.validate.WSGIWarning):
            response = s()
    else:
        response = s()
    assert 'BAR' in falcon.constants.COMBINED_METHODS
    assert response.status == falcon.HTTP_405
    assert response.status_code == 405
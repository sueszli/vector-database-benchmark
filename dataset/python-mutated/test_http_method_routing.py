from functools import wraps
import pytest
import falcon
import falcon.constants
import falcon.testing as testing
from _util import create_app
HTTP_METHODS = ['CONNECT', 'DELETE', 'GET', 'HEAD', 'OPTIONS', 'PATCH', 'POST', 'PUT', 'TRACE']
WEBDAV_METHODS = ['CHECKIN', 'CHECKOUT', 'COPY', 'LOCK', 'MKCOL', 'MOVE', 'PROPFIND', 'PROPPATCH', 'REPORT', 'UNCHECKIN', 'UNLOCK', 'UPDATE', 'VERSION-CONTROL']

@pytest.fixture
def stonewall():
    if False:
        return 10
    return Stonewall()

@pytest.fixture
def resource_things():
    if False:
        print('Hello World!')
    return ThingsResource()

@pytest.fixture
def resource_misc():
    if False:
        while True:
            i = 10
    return MiscResource()

@pytest.fixture
def resource_get_with_faulty_put():
    if False:
        return 10
    return GetWithFaultyPutResource()

@pytest.fixture
def client(asgi):
    if False:
        print('Hello World!')
    app = create_app(asgi)
    app.add_route('/stonewall', Stonewall())
    resource_things = ThingsResource()
    app.add_route('/things', resource_things)
    app.add_route('/things/{id}/stuff/{sid}', resource_things)
    resource_misc = MiscResource()
    app.add_route('/misc', resource_misc)
    resource_get_with_faulty_put = GetWithFaultyPutResource()
    app.add_route('/get_with_param/{param}', resource_get_with_faulty_put)
    return testing.TestClient(app)

class ThingsResource:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.called = False
        self.on_patch = {}

    def on_get(self, req, resp, sid, id):
        if False:
            return 10
        self.called = True
        (self.req, self.resp) = (req, resp)
        resp.status = falcon.HTTP_204

    def on_head(self, req, resp, id, sid):
        if False:
            return 10
        self.called = True
        (self.req, self.resp) = (req, resp)
        resp.status = falcon.HTTP_204

    def on_put(self, req, resp, id, sid):
        if False:
            i = 10
            return i + 15
        self.called = True
        (self.req, self.resp) = (req, resp)
        resp.status = falcon.HTTP_201

    def on_report(self, req, resp, id, sid):
        if False:
            i = 10
            return i + 15
        self.called = True
        (self.req, self.resp) = (req, resp)
        resp.status = falcon.HTTP_204

    def on_websocket(self, req, resp, id, sid):
        if False:
            return 10
        self.called = True

class Stonewall:
    pass

def capture(func):
    if False:
        i = 10
        return i + 15

    @wraps(func)
    def with_capture(*args, **kwargs):
        if False:
            print('Hello World!')
        self = args[0]
        self.called = True
        (self.req, self.resp) = args[1:]
        func(*args, **kwargs)
    return with_capture

def selfless_decorator(func):
    if False:
        while True:
            i = 10

    def faulty(req, resp, foo, bar):
        if False:
            print('Hello World!')
        pass
    return faulty

class MiscResource:

    def __init__(self):
        if False:
            print('Hello World!')
        self.called = False

    @capture
    def on_get(self, req, resp):
        if False:
            while True:
                i = 10
        resp.status = falcon.HTTP_204

    @capture
    def on_head(self, req, resp):
        if False:
            return 10
        resp.status = falcon.HTTP_204

    @capture
    def on_put(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        resp.status = falcon.HTTP_400

    @capture
    def on_patch(self, req, resp):
        if False:
            i = 10
            return i + 15
        pass

    def on_options(self, req, resp):
        if False:
            while True:
                i = 10
        resp.status = falcon.HTTP_204
        resp.set_header('allow', 'GET')

class GetWithFaultyPutResource:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.called = False

    @capture
    def on_get(self, req, resp):
        if False:
            while True:
                i = 10
        resp.status = falcon.HTTP_204

    def on_put(self, req, resp, param):
        if False:
            while True:
                i = 10
        raise TypeError()

class FaultyDecoratedResource:

    @selfless_decorator
    def on_get(self, req, resp):
        if False:
            i = 10
            return i + 15
        pass

class TestHttpMethodRouting:

    def test_get(self, client, resource_things):
        if False:
            return 10
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/42/stuff/57')
        assert response.status == falcon.HTTP_204
        assert resource_things.called

    def test_put(self, client, resource_things):
        if False:
            return 10
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/42/stuff/1337', method='PUT')
        assert response.status == falcon.HTTP_201
        assert resource_things.called

    def test_post_not_allowed(self, client, resource_things):
        if False:
            while True:
                i = 10
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/42/stuff/1337', method='POST')
        assert response.status == falcon.HTTP_405
        assert not resource_things.called

    def test_report(self, client, resource_things):
        if False:
            while True:
                i = 10
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/42/stuff/1337', method='REPORT')
        assert response.status == falcon.HTTP_204
        assert resource_things.called

    def test_misc(self, client, resource_misc):
        if False:
            i = 10
            return i + 15
        client.app.add_route('/misc', resource_misc)
        for method in ['GET', 'HEAD', 'PUT', 'PATCH']:
            resource_misc.called = False
            client.simulate_request(path='/misc', method=method)
            assert resource_misc.called
            assert resource_misc.req.method == method

    def test_methods_not_allowed_simple(self, client, stonewall):
        if False:
            print('Hello World!')
        client.app.add_route('/stonewall', stonewall)
        for method in ['GET', 'HEAD', 'PUT', 'PATCH']:
            response = client.simulate_request(path='/stonewall', method=method)
            assert response.status == falcon.HTTP_405

    def test_methods_not_allowed_complex(self, client, resource_things):
        if False:
            print('Hello World!')
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        for method in HTTP_METHODS + WEBDAV_METHODS:
            if method in ('GET', 'PUT', 'HEAD', 'OPTIONS', 'REPORT'):
                continue
            resource_things.called = False
            response = client.simulate_request(path='/things/84/stuff/65', method=method)
            assert not resource_things.called
            assert response.status == falcon.HTTP_405
            headers = response.headers
            assert headers['allow'] == 'GET, HEAD, PUT, REPORT, OPTIONS'

    def test_method_not_allowed_with_param(self, client, resource_get_with_faulty_put):
        if False:
            return 10
        client.app.add_route('/get_with_param/{param}', resource_get_with_faulty_put)
        for method in HTTP_METHODS + WEBDAV_METHODS:
            if method in ('GET', 'PUT', 'OPTIONS'):
                continue
            resource_get_with_faulty_put.called = False
            response = client.simulate_request(method=method, path='/get_with_param/bogus_param')
            assert not resource_get_with_faulty_put.called
            assert response.status == falcon.HTTP_405
            headers = response.headers
            assert headers['allow'] == 'GET, PUT, OPTIONS'

    def test_default_on_options(self, client, resource_things):
        if False:
            return 10
        client.app.add_route('/things', resource_things)
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/84/stuff/65', method='OPTIONS')
        assert response.status == falcon.HTTP_200
        headers = response.headers
        assert headers['allow'] == 'GET, HEAD, PUT, REPORT'

    def test_on_options(self, client):
        if False:
            print('Hello World!')
        response = client.simulate_request(path='/misc', method='OPTIONS')
        assert response.status == falcon.HTTP_204
        headers = response.headers
        assert headers['allow'] == 'GET'

    @pytest.mark.parametrize('method', falcon.constants._META_METHODS + ['SETECASTRONOMY'])
    @pytest.mark.filterwarnings('ignore:Unknown REQUEST_METHOD')
    def test_meta_and_others_disallowed(self, client, resource_things, method):
        if False:
            while True:
                i = 10
        client.app.add_route('/things/{id}/stuff/{sid}', resource_things)
        response = client.simulate_request(path='/things/42/stuff/1337', method='WEBSOCKET')
        assert response.status == falcon.HTTP_400
        assert not resource_things.called
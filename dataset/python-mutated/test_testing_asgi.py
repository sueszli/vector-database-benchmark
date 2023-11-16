import time
import pytest
import falcon
from falcon import testing
from . import _asgi_test_app

@pytest.mark.asyncio
async def test_asgi_request_event_emitter_hang():
    expected_elapsed_min = 1
    disconnect_at = time.time() + expected_elapsed_min
    emit = testing.ASGIRequestEventEmitter(disconnect_at=disconnect_at)
    start = time.time()
    while True:
        event = await emit()
        if not event.get('more_body', False):
            break
    elapsed = time.time() - start
    assert elapsed < 0.1
    start = time.time()
    await emit()
    elapsed = time.time() - start
    assert elapsed + 0.1 > expected_elapsed_min

@pytest.mark.asyncio
async def test_ignore_extra_asgi_events():
    collect = testing.ASGIResponseEventCollector()
    await collect({'type': 'http.response.start', 'status': 200})
    await collect({'type': 'http.response.body', 'more_body': False})
    await collect({'type': 'http.response.body'})
    assert len(collect.events) == 2

@pytest.mark.asyncio
async def test_invalid_asgi_events():
    collect = testing.ASGIResponseEventCollector()

    def make_event(headers=None, status=200):
        if False:
            return 10
        return {'type': 'http.response.start', 'headers': headers or [], 'status': status}
    with pytest.raises(TypeError):
        await collect({'type': 123})
    with pytest.raises(TypeError):
        headers = [('notbytes', b'bytes')]
        await collect(make_event(headers))
    with pytest.raises(TypeError):
        headers = [(b'bytes', 'notbytes')]
        await collect(make_event(headers))
    with pytest.raises(ValueError):
        headers = [(b'Content-Type', b'application/json')]
        await collect(make_event(headers))
    with pytest.raises(TypeError):
        await collect(make_event(status='200'))
    with pytest.raises(TypeError):
        await collect(make_event(status=200.1))
    with pytest.raises(TypeError):
        await collect({'type': 'http.response.body', 'body': 'notbytes'})
    with pytest.raises(TypeError):
        await collect({'type': 'http.response.body', 'more_body': ''})
    with pytest.raises(ValueError):
        await collect({'type': 'http.response.bod'})

def test_is_asgi_app_cls():
    if False:
        for i in range(10):
            print('nop')

    class Foo:

        @classmethod
        def class_meth(cls, scope, receive, send):
            if False:
                for i in range(10):
                    print('nop')
            pass
    assert testing.client._is_asgi_app(Foo.class_meth)

def test_cookies_jar():
    if False:
        while True:
            i = 10
    client = testing.TestClient(_asgi_test_app.application)
    response_one = client.simulate_get('/jars')
    response_two = client.simulate_post('/jars', cookies=response_one.cookies)
    assert response_two.status == falcon.HTTP_200

def test_create_scope_default_ua():
    if False:
        while True:
            i = 10
    default_ua = 'falcon-client/' + falcon.__version__
    scope = testing.create_scope()
    assert dict(scope['headers'])[b'user-agent'] == default_ua.encode()
    req = testing.create_asgi_req()
    assert req.user_agent == default_ua

def test_create_scope_default_ua_override():
    if False:
        i = 10
        return i + 15
    ua = 'curl/7.64.1'
    scope = testing.create_scope(headers={'user-agent': ua})
    assert dict(scope['headers'])[b'user-agent'] == ua.encode()
    req = testing.create_asgi_req(headers={'user-agent': ua})
    assert req.user_agent == ua

def test_create_scope_default_ua_modify_global():
    if False:
        for i in range(10):
            print('nop')
    default_ua = 'URL/Emacs Emacs/26.3 (x86_64-pc-linux-gnu)'
    prev_default = falcon.testing.helpers.DEFAULT_UA
    falcon.testing.helpers.DEFAULT_UA = default_ua
    try:
        req = testing.create_asgi_req()
        assert req.user_agent == default_ua
    finally:
        falcon.testing.helpers.DEFAULT_UA = prev_default

def test_missing_header_is_none():
    if False:
        while True:
            i = 10
    req = testing.create_asgi_req()
    assert req.auth is None

def test_immediate_disconnect():
    if False:
        i = 10
        return i + 15
    client = testing.TestClient(_asgi_test_app.application)
    with pytest.raises(ConnectionError):
        client.simulate_get('/', asgi_disconnect_ttl=0)
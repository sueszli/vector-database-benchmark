import json
import pytest
import falcon
from falcon import errors, media, testing
import falcon.asgi
from falcon.util.deprecation import DeprecatedWarning

def create_client(resource, handlers=None):
    if False:
        return 10
    app = falcon.asgi.App()
    app.add_route('/', resource)
    if handlers:
        app.resp_options.media_handlers.update(handlers)
    client = testing.TestClient(app, headers={'capture-resp-media': 'yes'})
    return client

class SimpleMediaResource:

    def __init__(self, document, media_type=falcon.MEDIA_JSON):
        if False:
            return 10
        self._document = document
        self._media_type = media_type

    async def on_get(self, req, resp):
        resp.content_type = self._media_type
        resp.media = self._document
        resp.status = falcon.HTTP_OK

@pytest.mark.parametrize('media_type', ['*/*', falcon.MEDIA_JSON, 'application/json; charset=utf-8'])
def test_json(media_type):
    if False:
        print('Hello World!')

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = media_type
            resp.media = {'something': True}
            body = await resp.render_body()
            assert json.loads(body.decode('utf-8')) == {'something': True}
    client = create_client(TestResource())
    client.simulate_get('/')

@pytest.mark.parametrize('document', ['', 'I am a ᴊꜱᴏɴ string.', ['♥', '♠', '♦', '♣'], {'message': '¡Hello Unicode! 😸'}, {'description': 'A collection of primitive Python type examples.', 'bool': False is not True and True is not False, 'dict': {'example': 'mapping'}, 'float': 1.0, 'int': 1337, 'list': ['a', 'sequence', 'of', 'items'], 'none': None, 'str': 'ASCII string', 'unicode': 'Hello Unicode! 😸'}])
def test_non_ascii_json_serialization(document):
    if False:
        i = 10
        return i + 15
    client = create_client(SimpleMediaResource(document))
    resp = client.simulate_get('/')
    assert resp.json == document

@pytest.mark.parametrize('media_type', [falcon.MEDIA_MSGPACK, 'application/msgpack; charset=utf-8', 'application/x-msgpack'])
def test_msgpack(media_type):
    if False:
        print('Hello World!')

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = media_type
            resp.media = {b'something': True}
            assert await resp.render_body() == b'\x81\xc4\tsomething\xc3'
            resp.media = {'something': True}
            body = await resp.render_body()
            assert body == b'\x81\xa9something\xc3'
            assert await resp.render_body() is body
    client = create_client(TestResource(), handlers={'application/msgpack': media.MessagePackHandler(), 'application/x-msgpack': media.MessagePackHandler()})
    client.simulate_get('/')

def test_custom_media_handler():
    if False:
        while True:
            i = 10

    class PythonRepresentation(media.BaseHandler):

        async def serialize_async(media, content_type):
            return repr(media).encode()

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = 'text/x-python-repr'
            resp.media = {'something': True}
            body = await resp.render_body()
            assert body == b"{'something': True}"
    client = create_client(TestResource(), handlers={'text/x-python-repr': PythonRepresentation()})
    client.simulate_get('/')

def test_unknown_media_type():
    if False:
        return 10

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = 'nope/json'
            resp.media = {'something': True}
            try:
                await resp.render_body()
            except Exception as ex:
                assert isinstance(ex, errors.HTTPUnsupportedMediaType)
                raise
    client = create_client(TestResource())
    result = client.simulate_get('/')
    assert result.status_code == 415

def test_default_media_type():
    if False:
        for i in range(10):
            print('nop')
    doc = {'something': True}

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = ''
            resp.media = {'something': True}
            body = await resp.render_body()
            assert json.loads(body.decode('utf-8')) == doc
            assert resp.content_type == 'application/json'
    client = create_client(TestResource())
    result = client.simulate_get('/')
    assert result.json == doc

@pytest.mark.parametrize('monkeypatch_resolver', [True, False])
def test_mimeparse_edgecases(monkeypatch_resolver):
    if False:
        for i in range(10):
            print('nop')
    doc = {'something': True}

    class TestResource:

        async def on_get(self, req, resp):
            resp.content_type = 'application/vnd.something'
            with pytest.raises(errors.HTTPUnsupportedMediaType):
                resp.media = {'something': False}
                await resp.render_body()
            resp.content_type = 'invalid'
            with pytest.raises(errors.HTTPUnsupportedMediaType):
                resp.media = {'something': False}
                await resp.render_body()
            for content_type in (None, '*/*'):
                resp.content_type = content_type
                resp.media = doc
    client = create_client(TestResource())
    handlers = client.app.resp_options.media_handlers
    if monkeypatch_resolver:

        def _resolve(media_type, default, raise_not_found=True):
            if False:
                i = 10
                return i + 15
            with pytest.warns(DeprecatedWarning, match='This undocumented method'):
                h = handlers.find_by_media_type(media_type, default, raise_not_found=raise_not_found)
            return (h, None, None)
        handlers._resolve = _resolve
    result = client.simulate_get('/')
    assert result.json == doc

def run_test(test_fn):
    if False:
        while True:
            i = 10
    doc = {'something': True}

    class TestResource:

        async def on_get(self, req, resp):
            await test_fn(resp)
            resp.text = None
            resp.data = None
            resp.media = doc
    client = create_client(TestResource())
    result = client.simulate_get('/')
    assert result.json == doc

class TestRenderBodyPrecedence:

    def test_text(self):
        if False:
            for i in range(10):
                print('nop')

        async def test(resp):
            resp.text = 'body'
            resp.data = b'data'
            resp.media = ['media']
            assert await resp.render_body() == b'body'
        run_test(test)

    def test_data(self):
        if False:
            i = 10
            return i + 15

        async def test(resp):
            resp.data = b'data'
            resp.media = ['media']
            assert await resp.render_body() == b'data'
        run_test(test)

    def test_data_masquerading_as_text(self):
        if False:
            i = 10
            return i + 15

        async def test(resp):
            resp.text = b'data'
            resp.media = ['media']
            assert await resp.render_body() == b'data'
        run_test(test)

    def test_media(self):
        if False:
            print('Hello World!')

        async def test(resp):
            resp.media = ['media']
            assert json.loads((await resp.render_body()).decode('utf-8')) == ['media']
        run_test(test)

def test_media_rendered_cached():
    if False:
        i = 10
        return i + 15

    async def test(resp):
        resp.media = {'foo': 'bar'}
        first = await resp.render_body()
        assert first is await resp.render_body()
        assert first is resp._media_rendered
        resp.media = 123
        assert first is not await resp.render_body()
    run_test(test)

def test_custom_render_body():
    if False:
        while True:
            i = 10

    class CustomResponse(falcon.asgi.Response):

        async def render_body(self):
            body = await super().render_body()
            if not self.content_type.startswith('text/plain'):
                return body
            if not body.endswith(b'\n'):
                return body + b'\n'
            return body

    class HelloResource:

        async def on_get(self, req, resp):
            resp.content_type = falcon.MEDIA_TEXT
            resp.text = 'Hello, World!'
    app = falcon.asgi.App(response_type=CustomResponse)
    app.add_route('/', HelloResource())
    resp = testing.simulate_get(app, '/')
    assert resp.headers['Content-Type'] == 'text/plain; charset=utf-8'
    assert resp.text == 'Hello, World!\n'
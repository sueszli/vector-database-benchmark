from functools import partial
import io
import json
import platform
import mujson
import pytest
import ujson
import falcon
from falcon import media, testing
from falcon.asgi.stream import BoundedStream
from falcon.util.deprecation import DeprecatedWarning
from _util import create_app
orjson = None
rapidjson = None
try:
    import rapidjson
except ImportError:
    pass
if platform.python_implementation() == 'CPython':
    try:
        import orjson
    except ImportError:
        pass
YEN = b'\xc2\xa5'
SERIALIZATION_PARAM_LIST = [(None, {'test': 'value'}, b'{"test":"value"}'), (partial(mujson.dumps, ensure_ascii=True), {'test': 'value'}, b'{"test":"value"}'), (ujson.dumps, {'test': 'value'}, b'{"test":"value"}'), (partial(lambda media, **kwargs: json.dumps([media, kwargs]), ensure_ascii=True), {'test': 'value'}, b'[{"test":"value"},{"ensure_ascii":true}]'), (None, {'yen': YEN.decode()}, b'{"yen":"' + YEN + b'"}')]
DESERIALIZATION_PARAM_LIST = [(None, b'[1, 2]', [1, 2]), (partial(json.loads, object_hook=lambda data: {k: v.upper() for (k, v) in data.items()}), b'{"key": "value"}', {'key': 'VALUE'}), (mujson.loads, b'{"test": "value"}', {'test': 'value'}), (ujson.loads, b'{"test": "value"}', {'test': 'value'})]
ALL_JSON_IMPL = [(json.dumps, json.loads), (partial(mujson.dumps, ensure_ascii=True), mujson.loads), (ujson.dumps, ujson.loads)]
if orjson:
    SERIALIZATION_PARAM_LIST += [(orjson.dumps, {'test': 'value'}, b'{"test":"value"}')]
    DESERIALIZATION_PARAM_LIST += [(orjson.loads, b'{"test": "value"}', {'test': 'value'})]
    ALL_JSON_IMPL += [(orjson.dumps, orjson.loads)]
if rapidjson:
    SERIALIZATION_PARAM_LIST += [(rapidjson.dumps, {'test': 'value'}, b'{"test":"value"}')]
    DESERIALIZATION_PARAM_LIST += [(rapidjson.loads, b'{"test": "value"}', {'test': 'value'})]
    ALL_JSON_IMPL += [(rapidjson.dumps, rapidjson.loads)]

@pytest.mark.parametrize('func, body, expected', SERIALIZATION_PARAM_LIST)
def test_serialization(asgi, func, body, expected):
    if False:
        i = 10
        return i + 15
    handler = media.JSONHandler(dumps=func)
    args = (body, b'application/javacript')
    if asgi:
        result = falcon.async_to_sync(handler.serialize_async, *args)
    else:
        result = handler.serialize(*args)
    assert result.replace(b' ', b'') == expected

@pytest.mark.parametrize('func, body, expected', DESERIALIZATION_PARAM_LIST)
def test_deserialization(asgi, func, body, expected):
    if False:
        i = 10
        return i + 15
    handler = media.JSONHandler(loads=func)
    args = ['application/javacript', len(body)]
    if asgi:
        s = BoundedStream(testing.ASGIRequestEventEmitter(body))
        args.insert(0, s)
        result = falcon.async_to_sync(handler.deserialize_async, *args)
    else:
        args.insert(0, io.BytesIO(body))
        result = handler.deserialize(*args)
    assert result == expected

@pytest.mark.parametrize('dumps, loads', ALL_JSON_IMPL)
@pytest.mark.parametrize('subclass', (True, False))
def test_full_app(asgi, dumps, loads, subclass):
    if False:
        i = 10
        return i + 15
    if subclass:

        class JSONHandlerSubclass(media.JSONHandler):
            pass
        handler = JSONHandlerSubclass(dumps=dumps, loads=loads)
        assert handler._serialize_sync is None
        assert handler._deserialize_sync is None
    else:
        handler = media.JSONHandler(dumps=dumps, loads=loads)
        assert handler._serialize_sync is not None
        assert handler._deserialize_sync is not None
    app = create_app(asgi)
    app.req_options.media_handlers[falcon.MEDIA_JSON] = handler
    app.resp_options.media_handlers[falcon.MEDIA_JSON] = handler
    if asgi:

        class Resp:

            async def on_get(self, req, res):
                res.media = await req.get_media()
    else:

        class Resp:

            def on_get(self, req, res):
                if False:
                    print('Hello World!')
                res.media = req.get_media()
    app.add_route('/go', Resp())
    data = {'foo': 123, 'bar': [2, 3], 'baz': {'x': 'y'}}
    res = testing.simulate_get(app, '/go', json=data)
    assert res.json == data

@pytest.mark.parametrize('monkeypatch_resolver', [True, False])
@pytest.mark.parametrize('handler_mt', ['application/json', 'application/json; answer=42'])
def test_deserialization_raises(asgi, handler_mt, monkeypatch_resolver):
    if False:
        i = 10
        return i + 15
    app = create_app(asgi)

    class SuchException(Exception):
        pass

    class FaultyHandler(media.BaseHandler):

        def deserialize(self, stream, content_type, content_length):
            if False:
                print('Hello World!')
            raise SuchException('Wow such error.')

        def deserialize_async(self, stream, content_type, content_length):
            if False:
                while True:
                    i = 10
            raise SuchException('Wow such error.')

        def serialize(self, media, content_type):
            if False:
                while True:
                    i = 10
            raise SuchException('Wow such error.')
    handlers = media.Handlers({handler_mt: FaultyHandler()})
    if monkeypatch_resolver:

        def _resolve(media_type, default, raise_not_found=True):
            if False:
                while True:
                    i = 10
            with pytest.warns(DeprecatedWarning, match='This undocumented method'):
                h = handlers.find_by_media_type(media_type, default, raise_not_found=raise_not_found)
            return (h, None, None)
        handlers._resolve = _resolve
    app.req_options.media_handlers = handlers
    app.resp_options.media_handlers = handlers

    class Resource:

        def on_get(self, req, resp):
            if False:
                print('Hello World!')
            resp.media = {}

        def on_post(self, req, resp):
            if False:
                print('Hello World!')
            req.media

    class ResourceAsync:

        async def on_get(self, req, resp):
            resp.media = {}

        async def on_post(self, req, resp):
            await req.media
    app.add_route('/', ResourceAsync() if asgi else Resource())
    app._error_handlers.clear()
    with pytest.raises(SuchException):
        testing.simulate_get(app, '/')
    with pytest.raises(SuchException):
        testing.simulate_post(app, '/', json={})

def test_sync_methods_not_overridden(asgi):
    if False:
        return 10
    app = create_app(asgi)

    class FaultyHandler(media.BaseHandler):
        pass
    handlers = media.Handlers({'application/json': FaultyHandler()})
    app.req_options.media_handlers = handlers
    app.resp_options.media_handlers = handlers

    class Resource:

        def on_get(self, req, resp):
            if False:
                i = 10
                return i + 15
            resp.media = {}

        def on_post(self, req, resp):
            if False:
                i = 10
                return i + 15
            req.media

    class ResourceAsync:

        async def on_get(self, req, resp):
            resp.media = {}

        async def on_post(self, req, resp):
            await req.get_media()
    app.add_route('/', ResourceAsync() if asgi else Resource())
    result = testing.simulate_get(app, '/', headers={'Accept': 'text/xml'})
    assert result.status_code == 500
    result = testing.simulate_post(app, '/', json={}, headers={'Accept': 'text/xml'})
    assert result.status_code == 500

def test_async_methods_not_overridden():
    if False:
        for i in range(10):
            print('nop')
    app = create_app(asgi=True)

    class SimpleHandler(media.BaseHandler):

        def serialize(self, media, content_type):
            if False:
                while True:
                    i = 10
            return json.dumps(media).encode()

        def deserialize(self, stream, content_type, content_length):
            if False:
                i = 10
                return i + 15
            return json.load(stream)
    handlers = media.Handlers({'application/json': SimpleHandler()})
    app.req_options.media_handlers = handlers
    app.resp_options.media_handlers = handlers

    class ResourceAsync:

        async def on_post(self, req, resp):
            resp.media = await req.get_media()
    app.add_route('/', ResourceAsync())
    doc = {'event': 'serialized'}
    result = testing.simulate_post(app, '/', json=doc)
    assert result.status_code == 200
    assert result.json == doc

def test_async_handler_returning_none():
    if False:
        return 10
    app = create_app(asgi=True)

    class SimpleHandler(media.BaseHandler):

        def serialize(self, media, content_type):
            if False:
                for i in range(10):
                    print('nop')
            return json.dumps(media).encode()

        def deserialize(self, stream, content_type, content_length):
            if False:
                return 10
            return None
    handlers = media.Handlers({'application/json': SimpleHandler()})
    app.req_options.media_handlers = handlers
    app.resp_options.media_handlers = handlers

    class ResourceAsync:

        async def on_post(self, req, resp):
            resp.media = [await req.get_media()]
    app.add_route('/', ResourceAsync())
    doc = {'event': 'serialized'}
    result = testing.simulate_post(app, '/', json=doc)
    assert result.status_code == 200
    assert result.json == [None]

@pytest.mark.parametrize('monkeypatch_resolver', [True, False])
def test_json_err_no_handler(asgi, monkeypatch_resolver):
    if False:
        i = 10
        return i + 15
    app = create_app(asgi)
    handlers = media.Handlers({falcon.MEDIA_URLENCODED: media.URLEncodedFormHandler()})
    if monkeypatch_resolver:

        def _resolve(media_type, default, raise_not_found=True):
            if False:
                i = 10
                return i + 15
            with pytest.warns(DeprecatedWarning, match='This undocumented method'):
                h = handlers.find_by_media_type(media_type, default, raise_not_found=raise_not_found)
            return (h, None, None)
        handlers._resolve = _resolve
    app.req_options.media_handlers = handlers
    app.resp_options.media_handlers = handlers

    class Resource:

        def on_get(self, req, resp):
            if False:
                while True:
                    i = 10
            raise falcon.HTTPForbidden()
    app.add_route('/', Resource())
    result = testing.simulate_get(app, '/')
    assert result.status_code == 403
    assert result.json == falcon.HTTPForbidden().to_dict()

class TestBaseHandler:

    def test_defaultError(self):
        if False:
            return 10
        h = media.BaseHandler()

        def test(call):
            if False:
                return 10
            with pytest.raises(NotImplementedError) as e:
                call()
            assert e.value.args == ()
        test(lambda : h.serialize({}, 'my-type'))
        test(lambda : h.deserialize('', 'my-type', 0))

    def test_json(self):
        if False:
            while True:
                i = 10
        h = media.BaseHandler()
        with pytest.raises(NotImplementedError, match='The JSON media handler requires'):
            h.serialize({}, falcon.MEDIA_JSON)
        with pytest.raises(NotImplementedError, match='The JSON media handler requires'):
            h.deserialize('', falcon.MEDIA_JSON, 0)
        with pytest.raises(NotImplementedError, match='The JSON media handler requires'):
            h.serialize({}, falcon.MEDIA_JSON + '; charset=UTF-8')
        with pytest.raises(NotImplementedError, match='The JSON media handler requires'):
            h.deserialize('', falcon.MEDIA_JSON + '; charset=UTF-8', 0)
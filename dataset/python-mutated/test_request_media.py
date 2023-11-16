import json
import pytest
import falcon
from falcon import errors, media, testing, util
from _util import create_app

def create_client(asgi, handlers=None, resource=None):
    if False:
        while True:
            i = 10
    if not resource:
        resource = testing.SimpleTestResourceAsync() if asgi else testing.SimpleTestResource()
    app = create_app(asgi)
    app.add_route('/', resource)
    if handlers:
        app.req_options.media_handlers.update(handlers)
    client = testing.TestClient(app, headers={'capture-req-media': 'yes'})
    client.resource = resource
    return client

@pytest.fixture()
def client(asgi):
    if False:
        for i in range(10):
            print('nop')
    return create_client(asgi)

class ResourceCachedMedia:

    def on_post(self, req, resp, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.captured_req_media = req.media
        assert self.captured_req_media is req.get_media()

class ResourceCachedMediaAsync:

    async def on_post(self, req, resp, **kwargs):
        self.captured_req_media = await req.get_media()
        assert self.captured_req_media is await req.get_media()

class ResourceInvalidMedia:

    def __init__(self, expected_error):
        if False:
            print('Hello World!')
        self._expected_error = expected_error

    def on_post(self, req, resp, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(self._expected_error) as error:
            req.media
        self.captured_error = error

class ResourceInvalidMediaAsync:

    def __init__(self, expected_error):
        if False:
            for i in range(10):
                print('nop')
        self._expected_error = expected_error

    async def on_post(self, req, resp, **kwargs):
        with pytest.raises(self._expected_error) as error:
            await req.get_media()
        self.captured_error = error

@pytest.mark.parametrize('media_type', [None, '*/*', 'application/json', 'application/json; charset=utf-8'])
def test_json(client, media_type):
    if False:
        while True:
            i = 10
    expected_body = b'{"something": true}'
    headers = {'Content-Type': media_type}
    client.simulate_post('/', body=expected_body, headers=headers)
    media = client.resource.captured_req_media
    assert media is not None
    assert media.get('something') is True

@pytest.mark.parametrize('media_type', ['application/msgpack', 'application/msgpack; charset=utf-8', 'application/x-msgpack'])
def test_msgpack(asgi, media_type):
    if False:
        while True:
            i = 10
    client = create_client(asgi, {'application/msgpack': media.MessagePackHandler(), 'application/x-msgpack': media.MessagePackHandler()})
    headers = {'Content-Type': media_type}
    expected_body = b'\x81\xc4\tsomething\xc3'
    assert client.simulate_post('/', body=expected_body, headers=headers).status_code == 200
    req_media = client.resource.captured_req_media
    assert req_media.get(b'something') is True
    expected_body = b'\x81\xa9something\xc3'
    assert client.simulate_post('/', body=expected_body, headers=headers).status_code == 200
    req_media = client.resource.captured_req_media
    assert req_media.get('something') is True

@pytest.mark.parametrize('media_type', ['nope/json'])
def test_unknown_media_type(asgi, media_type):
    if False:
        print('Hello World!')
    client = _create_client_invalid_media(asgi, errors.HTTPUnsupportedMediaType)
    headers = {'Content-Type': media_type}
    assert client.simulate_post('/', body=b'something', headers=headers).status_code == 200
    title_msg = '415 Unsupported Media Type'
    description_msg = '{} is an unsupported media type.'.format(media_type)
    assert client.resource.captured_error.value.title == title_msg
    assert client.resource.captured_error.value.description == description_msg

@pytest.mark.parametrize('media_type', ['application/json', 'application/msgpack'])
def test_empty_body(asgi, media_type):
    if False:
        for i in range(10):
            print('nop')
    client = _create_client_invalid_media(asgi, errors.HTTPBadRequest, {'application/msgpack': media.MessagePackHandler(), 'application/json': media.JSONHandler()})
    headers = {'Content-Type': media_type}
    assert client.simulate_post('/', headers=headers).status_code == 200
    assert 'Could not parse an empty' in client.resource.captured_error.value.description
    assert isinstance(client.resource.captured_error.value, errors.MediaNotFoundError)

def test_invalid_json(asgi):
    if False:
        i = 10
        return i + 15
    client = _create_client_invalid_media(asgi, errors.HTTPBadRequest)
    expected_body = '{'
    headers = {'Content-Type': 'application/json'}
    assert client.simulate_post('/', body=expected_body, headers=headers).status_code == 200
    assert 'Could not parse JSON body' in client.resource.captured_error.value.description
    assert isinstance(client.resource.captured_error.value, errors.MediaMalformedError)
    try:
        json.loads(expected_body)
    except Exception as e:
        assert type(client.resource.captured_error.value.__cause__) is type(e)
        assert str(client.resource.captured_error.value.__cause__) == str(e)

def test_invalid_msgpack(asgi):
    if False:
        i = 10
        return i + 15
    import msgpack
    handlers = {'application/msgpack': media.MessagePackHandler()}
    client = _create_client_invalid_media(asgi, errors.HTTPBadRequest, handlers=handlers)
    expected_body = '/////////////////////'
    headers = {'Content-Type': 'application/msgpack'}
    assert client.simulate_post('/', body=expected_body, headers=headers).status_code == 200
    desc = 'Could not parse MessagePack body - unpack(b) received extra data.'
    assert client.resource.captured_error.value.description == desc
    assert isinstance(client.resource.captured_error.value, errors.MediaMalformedError)
    try:
        msgpack.unpackb(expected_body.encode('utf-8'))
    except Exception as e:
        assert type(client.resource.captured_error.value.__cause__) is type(e)
        assert str(client.resource.captured_error.value.__cause__) == str(e)

class NopeHandler(media.BaseHandler):

    def serialize(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def deserialize(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass
    exhaust_stream = True

def test_complete_consumption(asgi):
    if False:
        return 10
    client = create_client(asgi, {'nope/nope': NopeHandler()})
    body = b'{"something": "abracadabra"}'
    headers = {'Content-Type': 'nope/nope'}
    assert client.simulate_post('/', body=body, headers=headers).status_code == 200
    req_media = client.resource.captured_req_media
    assert req_media is None
    req_bounded_stream = client.resource.captured_req.bounded_stream
    assert req_bounded_stream.eof

@pytest.mark.parametrize('payload', [False, 0, 0.0, '', [], {}])
def test_empty_json_media(asgi, payload):
    if False:
        i = 10
        return i + 15
    resource = ResourceCachedMediaAsync() if asgi else ResourceCachedMedia()
    client = create_client(asgi, resource=resource)
    assert client.simulate_post('/', json=payload).status_code == 200
    assert resource.captured_req_media == payload

def test_null_json_media(client):
    if False:
        i = 10
        return i + 15
    assert client.simulate_post('/', body='null', headers={'Content-Type': 'application/json'}).status_code == 200
    assert client.resource.captured_req_media is None

def _create_client_invalid_media(asgi, error_type, handlers=None):
    if False:
        print('Hello World!')
    resource_type = ResourceInvalidMediaAsync if asgi else ResourceInvalidMedia
    resource = resource_type(error_type)
    return create_client(asgi, handlers=handlers, resource=resource)

class FallBack:

    def on_get(self, req, res):
        if False:
            print('Hello World!')
        res.media = req.get_media('fallback')

class FallBackAsync:

    async def on_get(self, req, res):
        res.media = await req.get_media('fallback')

def test_fallback(asgi):
    if False:
        i = 10
        return i + 15
    client = create_client(asgi, resource=FallBackAsync() if asgi else FallBack())
    res = client.simulate_get('/')
    assert res.status_code == 200
    assert res.json == 'fallback'

@pytest.mark.parametrize('exhaust_stream', (True, False))
@pytest.mark.parametrize('body', (True, False))
def test_fallback_not_for_error_body(asgi, exhaust_stream, body):
    if False:
        print('Hello World!')
    js = media.JSONHandler()
    js.exhaust_stream = exhaust_stream
    client = create_client(asgi, resource=FallBackAsync() if asgi else FallBack(), handlers={'application/json': js})
    res = client.simulate_get('/', body=b'{' if body else '')
    if body:
        assert res.status_code == 400
        assert 'Could not parse JSON body' in res.json['description']
    else:
        assert res.status_code == 200

def test_fallback_does_not_override_media_default(asgi):
    if False:
        while True:
            i = 10
    client = create_client(asgi, resource=FallBackAsync() if asgi else FallBack())
    res = client.simulate_get('/', headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert res.status_code == 200
    assert res.text == '{}'

async def _check_error(req, isasync):
    (err, err2) = (None, None)
    try:
        await req.media if isasync else req.media
    except Exception as e:
        err = e
    try:
        await req.get_media() if isasync else req.get_media()
    except Exception as e:
        err2 = e
    assert err is not None
    assert err2 is not None
    assert err2 is err
    if isinstance(err, errors.MediaMalformedError):
        assert err2.__cause__ is err.__cause__
    obj = {}
    if req.get_param_as_bool('empty'):
        res = await req.get_media(obj) if isasync else req.get_media(obj)
        assert res is obj
        err3 = None
        try:
            (await req.media if isasync else req.media) is obj
        except Exception as e:
            err3 = e
        assert err3 is err
    else:
        err3 = None
        try:
            await req.get_media(obj) if isasync else req.get_media(obj)
        except Exception as e:
            err3 = e
        assert err3 is err
    raise errors.HTTPError(falcon.HTTP_IM_A_TEAPOT)

class RepeatedError:

    def on_get(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        util.async_to_sync(_check_error, req, False)

class RepeatedErrorAsync:

    async def on_get(self, req, resp):
        await _check_error(req, True)

@pytest.mark.parametrize('body', ('{', ''))
def test_repeated_error(asgi, body):
    if False:
        return 10
    client = create_client(asgi, resource=RepeatedErrorAsync() if asgi else RepeatedError())
    res = client.simulate_get('/', body=body, params={'empty': not bool(body)})
    assert res.status == falcon.HTTP_IM_A_TEAPOT

def test_error_after_first_default(asgi):
    if False:
        while True:
            i = 10

    async def _check_error(req, isasync):
        assert await req.get_media(42) if isasync else req.get_media(42) == 42
        try:
            await req.get_media() if isasync else req.get_media()
        except falcon.MediaNotFoundError:
            raise falcon.HTTPStatus(falcon.HTTP_749)
        raise falcon.HTTPStatus(falcon.HTTP_703)

    class Res:

        def on_get(self, req, resp):
            if False:
                while True:
                    i = 10
            util.async_to_sync(_check_error, req, False)

    class ResAsync:

        async def on_get(self, req, resp):
            await _check_error(req, True)
    client = create_client(asgi, resource=ResAsync() if asgi else Res())
    res = client.simulate_get('/', body='')
    assert res.status == falcon.HTTP_749
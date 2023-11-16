import functools
import json
import pytest
import falcon
from falcon import testing
from _util import create_app, create_resp

@pytest.fixture
def wrapped_resource_aware():
    if False:
        while True:
            i = 10
    return ClassResourceWithAwareHooks()

@pytest.fixture
def client(asgi):
    if False:
        print('Hello World!')
    app = create_app(asgi)
    resource = WrappedRespondersResourceAsync() if asgi else WrappedRespondersResource()
    app.add_route('/', resource)
    return testing.TestClient(app)

def validate_output(req, resp, resource):
    if False:
        i = 10
        return i + 15
    assert resource
    raise falcon.HTTPError(falcon.HTTP_723, title='Tricky')

def serialize_body(req, resp, resource):
    if False:
        for i in range(10):
            print('nop')
    assert resource
    body = resp.text
    if body is not None:
        resp.text = json.dumps(body)
    else:
        resp.text = 'Nothing to see here. Move along.'

async def serialize_body_async(*args):
    return serialize_body(*args)

def fluffiness(req, resp, resource, animal=''):
    if False:
        return 10
    assert resource
    resp.text = 'fluffy'
    if animal:
        resp.set_header('X-Animal', animal)

class ResourceAwareFluffiness:

    def __call__(self, req, resp, resource):
        if False:
            i = 10
            return i + 15
        fluffiness(req, resp, resource)

def cuteness(req, resp, resource, check, postfix=' and cute'):
    if False:
        for i in range(10):
            print('nop')
    assert resource
    if resp.text == check:
        resp.text += postfix

def resource_aware_cuteness(req, resp, resource):
    if False:
        return 10
    assert resource
    cuteness(req, resp, resource, 'fluffy')

class Smartness:

    def __call__(self, req, resp, resource):
        if False:
            while True:
                i = 10
        assert resource
        if resp.text:
            resp.text += ' and smart'
        else:
            resp.text = 'smart'

def things_in_the_head(header, value, req, resp, resource):
    if False:
        for i in range(10):
            print('nop')
    assert resource
    resp.set_header(header, value)
bunnies_in_the_head = functools.partial(things_in_the_head, 'X-Bunnies', 'fluffy')
cuteness_in_the_head = functools.partial(things_in_the_head, 'X-Cuteness', 'cute')

def fluffiness_in_the_head(req, resp, resource, value='fluffy'):
    if False:
        while True:
            i = 10
    resp.set_header('X-Fluffiness', value)

class WrappedRespondersResource:

    @falcon.after(serialize_body)
    @falcon.after(validate_output)
    def on_get(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        self.req = req
        self.resp = resp

    @falcon.after(serialize_body)
    def on_put(self, req, resp):
        if False:
            print('Hello World!')
        self.req = req
        self.resp = resp
        resp.text = {'animal': 'falcon'}

    @falcon.after(Smartness())
    def on_post(self, req, resp):
        if False:
            print('Hello World!')
        pass

class WrappedRespondersResourceAsync:

    @falcon.after(serialize_body_async)
    @falcon.after(validate_output, is_async=False)
    async def on_get(self, req, resp):
        self.req = req
        self.resp = resp

    @falcon.after(serialize_body_async, is_async=True)
    async def on_put(self, req, resp):
        self.req = req
        self.resp = resp
        resp.text = {'animal': 'falcon'}

    @falcon.after(Smartness())
    async def on_post(self, req, resp):
        pass

@falcon.after(cuteness, 'fluffy', postfix=' and innocent')
@falcon.after(fluffiness, 'kitten')
class WrappedClassResource:
    on_post = False

    def __init__(self):
        if False:
            while True:
                i = 10
        self.on_patch = []

    def on_get(self, req, resp):
        if False:
            return 10
        self.req = req
        self.resp = resp

    @falcon.after(fluffiness_in_the_head)
    @falcon.after(cuteness_in_the_head)
    def on_head(self, req, resp):
        if False:
            i = 10
            return i + 15
        self.req = req
        self.resp = resp

class WrappedClassResourceChild(WrappedClassResource):

    def on_head(self, req, resp):
        if False:
            while True:
                i = 10
        super(WrappedClassResourceChild, self).on_head(req, resp)

class ClassResourceWithURIFields:

    @falcon.after(fluffiness_in_the_head, 'fluffy')
    def on_get(self, req, resp, field1, field2):
        if False:
            for i in range(10):
                print('nop')
        self.fields = (field1, field2)

class ClassResourceWithURIFieldsAsync:

    @falcon.after(fluffiness_in_the_head, 'fluffy')
    async def on_get(self, req, resp, field1, field2):
        self.fields = (field1, field2)

class ClassResourceWithURIFieldsChild(ClassResourceWithURIFields):

    def on_get(self, req, resp, field1, field2):
        if False:
            while True:
                i = 10
        super(ClassResourceWithURIFieldsChild, self).on_get(req, resp, field1, field2=field2)

@falcon.after(resource_aware_cuteness)
class ClassResourceWithAwareHooks:
    on_delete = False
    hook_as_class = ResourceAwareFluffiness()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.on_patch = []

    @falcon.after(fluffiness)
    def on_get(self, req, resp):
        if False:
            i = 10
            return i + 15
        self._capture(req, resp)

    @falcon.after(fluffiness)
    def on_head(self, req, resp):
        if False:
            return 10
        self._capture(req, resp)

    @falcon.after(hook_as_class)
    def on_put(self, req, resp):
        if False:
            print('Hello World!')
        self._capture(req, resp)

    @falcon.after(hook_as_class.__call__)
    def on_post(self, req, resp):
        if False:
            return 10
        self._capture(req, resp)

    def _capture(self, req, resp):
        if False:
            while True:
                i = 10
        self.req = req
        self.resp = resp

def test_output_validator(client):
    if False:
        print('Hello World!')
    result = client.simulate_get()
    assert result.status_code == 723
    assert result.text == json.dumps({'title': 'Tricky'})

def test_serializer(client):
    if False:
        for i in range(10):
            print('nop')
    result = client.simulate_put()
    assert result.text == json.dumps({'animal': 'falcon'})

def test_hook_as_callable_class(client):
    if False:
        print('Hello World!')
    result = client.simulate_post()
    assert 'smart' == result.text

@pytest.mark.parametrize('resource', [ClassResourceWithURIFields(), ClassResourceWithURIFieldsChild()])
def test_resource_with_uri_fields(client, resource):
    if False:
        for i in range(10):
            print('nop')
    client.app.add_route('/{field1}/{field2}', resource)
    result = client.simulate_get('/82074/58927')
    assert result.status_code == 200
    assert result.headers['X-Fluffiness'] == 'fluffy'
    assert 'X-Cuteness' not in result.headers
    assert resource.fields == ('82074', '58927')

def test_resource_with_uri_fields_async():
    if False:
        while True:
            i = 10
    app = create_app(asgi=True)
    resource = ClassResourceWithURIFieldsAsync()
    app.add_route('/{field1}/{field2}', resource)
    result = testing.simulate_get(app, '/a/b')
    assert result.status_code == 200
    assert result.headers['X-Fluffiness'] == 'fluffy'
    assert resource.fields == ('a', 'b')

    async def test_direct():
        resource = ClassResourceWithURIFieldsAsync()
        req = testing.create_asgi_req()
        resp = create_resp(True)
        await resource.on_get(req, resp, '1', '2')
        assert resource.fields == ('1', '2')
    falcon.async_to_sync(test_direct)

@pytest.mark.parametrize('resource', [WrappedClassResource(), WrappedClassResourceChild()])
def test_wrapped_resource(client, resource):
    if False:
        while True:
            i = 10
    client.app.add_route('/wrapped', resource)
    result = client.simulate_get('/wrapped')
    assert result.status_code == 200
    assert result.text == 'fluffy and innocent'
    assert result.headers['X-Animal'] == 'kitten'
    result = client.simulate_head('/wrapped')
    assert result.status_code == 200
    assert result.headers['X-Fluffiness'] == 'fluffy'
    assert result.headers['X-Cuteness'] == 'cute'
    assert result.headers['X-Animal'] == 'kitten'
    result = client.simulate_post('/wrapped')
    assert result.status_code == 405
    result = client.simulate_patch('/wrapped')
    assert result.status_code == 405
    result = client.simulate_options('/wrapped')
    assert result.status_code == 200
    assert not result.text
    assert 'X-Animal' not in result.headers

def test_wrapped_resource_with_hooks_aware_of_resource(client, wrapped_resource_aware):
    if False:
        print('Hello World!')
    client.app.add_route('/wrapped_aware', wrapped_resource_aware)
    expected = 'fluffy and cute'
    result = client.simulate_get('/wrapped_aware')
    assert result.status_code == 200
    assert expected == result.text
    for test in (client.simulate_head, client.simulate_put, client.simulate_post):
        result = test(path='/wrapped_aware')
        assert result.status_code == 200
        assert wrapped_resource_aware.resp.text == expected
    result = client.simulate_patch('/wrapped_aware')
    assert result.status_code == 405
    result = client.simulate_options('/wrapped_aware')
    assert result.status_code == 200
    assert not result.text

class ResourceAwareGameHook:
    VALUES = ('rock', 'scissors', 'paper')

    @classmethod
    def __call__(cls, req, resp, resource):
        if False:
            while True:
                i = 10
        assert resource
        assert resource.seed in cls.VALUES
        assert resp.text == 'Responder called.'
        header = resp.get_header('X-Hook-Game')
        values = header.split(', ') if header else []
        if values:
            last = cls.VALUES.index(values[-1])
            values.append(cls.VALUES[(last + 1) % len(cls.VALUES)])
        else:
            values.append(resource.seed)
        resp.set_header('X-Hook-Game', ', '.join(values))
_game_hook = ResourceAwareGameHook()

@falcon.after(_game_hook)
@falcon.after(_game_hook)
class HandGame:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.seed = None

    @falcon.after(_game_hook)
    def on_put(self, req, resp):
        if False:
            return 10
        self.seed = req.media
        resp.text = 'Responder called.'

    @falcon.after(_game_hook)
    def on_get_once(self, req, resp):
        if False:
            i = 10
            return i + 15
        resp.text = 'Responder called.'

    @falcon.after(_game_hook)
    @falcon.after(_game_hook)
    def on_get_twice(self, req, resp):
        if False:
            i = 10
            return i + 15
        resp.text = 'Responder called.'

    @falcon.after(_game_hook)
    @falcon.after(_game_hook)
    @falcon.after(_game_hook)
    def on_get_thrice(self, req, resp):
        if False:
            while True:
                i = 10
        resp.text = 'Responder called.'

@pytest.fixture
def game_client():
    if False:
        i = 10
        return i + 15
    app = falcon.App()
    resource = HandGame()
    app.add_route('/seed', resource)
    app.add_route('/once', resource, suffix='once')
    app.add_route('/twice', resource, suffix='twice')
    app.add_route('/thrice', resource, suffix='thrice')
    return testing.TestClient(app)

@pytest.mark.parametrize('seed,uri,expected', [('paper', '/once', 'paper, rock, scissors'), ('scissors', '/twice', 'scissors, paper, rock, scissors'), ('rock', '/thrice', 'rock, scissors, paper, rock, scissors'), ('paper', '/thrice', 'paper, rock, scissors, paper, rock')])
def test_after_hooks_on_suffixed_resource(game_client, seed, uri, expected):
    if False:
        return 10
    game_client.simulate_put('/seed', json=seed)
    resp = game_client.simulate_get(uri)
    assert resp.status_code == 200
    assert resp.headers['X-Hook-Game'] == expected
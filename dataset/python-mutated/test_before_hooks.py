import functools
import io
import json
import pytest
import falcon
import falcon.testing as testing
from _util import create_app, create_resp, disable_asgi_non_coroutine_wrapping

def validate(req, resp, resource, params):
    if False:
        while True:
            i = 10
    assert resource
    raise falcon.HTTPBadRequest(title='Invalid thing', description='Your thing was not formatted correctly.')

def validate_param(req, resp, resource, params, param_name, maxval=100):
    if False:
        for i in range(10):
            print('nop')
    assert resource
    limit = req.get_param_as_int(param_name)
    if limit and int(limit) > maxval:
        msg = '{0} must be <= {1}'.format(param_name, maxval)
        raise falcon.HTTPBadRequest(title='Out of Range', description=msg)

async def validate_param_async(*args, **kwargs):
    validate_param(*args, **kwargs)

class ResourceAwareValidateParam:

    def __call__(self, req, resp, resource, params):
        if False:
            print('Hello World!')
        assert resource
        validate_param(req, resp, resource, params, 'limit')

def validate_field(req, resp, resource, params, field_name='test'):
    if False:
        print('Hello World!')
    assert resource
    try:
        params[field_name] = int(params[field_name])
    except ValueError:
        raise falcon.HTTPBadRequest()

def parse_body(req, resp, resource, params):
    if False:
        return 10
    assert resource
    length = req.content_length
    if length:
        params['doc'] = json.load(io.TextIOWrapper(req.bounded_stream, 'utf-8'))

async def parse_body_async(req, resp, resource, params):
    assert resource
    length = req.content_length
    if length:
        data = await req.bounded_stream.read()
        params['doc'] = json.loads(data.decode('utf-8'))

def bunnies(req, resp, resource, params):
    if False:
        for i in range(10):
            print('nop')
    assert resource
    params['bunnies'] = 'fuzzy'

def frogs(req, resp, resource, params):
    if False:
        while True:
            i = 10
    assert resource
    if 'bunnies' in params:
        params['bunnies'] = 'fluffy'
    params['frogs'] = 'not fluffy'

class Fish:

    def __call__(self, req, resp, resource, params):
        if False:
            print('Hello World!')
        assert resource
        params['fish'] = 'slippery'

    def hook(self, req, resp, resource, params):
        if False:
            while True:
                i = 10
        assert resource
        params['fish'] = 'wet'

def things_in_the_head(header, value, req, resp, resource, params):
    if False:
        for i in range(10):
            print('nop')
    resp.set_header(header, value)
bunnies_in_the_head = functools.partial(things_in_the_head, 'X-Bunnies', 'fluffy')
frogs_in_the_head = functools.partial(things_in_the_head, 'X-Frogs', 'not fluffy')

class WrappedRespondersResource:

    @falcon.before(validate_param, 'limit', 100)
    def on_get(self, req, resp):
        if False:
            print('Hello World!')
        self.req = req
        self.resp = resp

    @falcon.before(validate)
    def on_put(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        self.req = req
        self.resp = resp

class WrappedRespondersResourceChild(WrappedRespondersResource):

    @falcon.before(validate_param, 'x', maxval=1000)
    def on_get(self, req, resp):
        if False:
            print('Hello World!')
        pass

    def on_put(self, req, resp):
        if False:
            while True:
                i = 10
        super(WrappedRespondersResourceChild, self).on_put(req, resp)

class WrappedRespondersBodyParserResource:

    @falcon.before(validate_param, 'limit', 100)
    @falcon.before(parse_body)
    def on_get(self, req, resp, doc=None):
        if False:
            for i in range(10):
                print('nop')
        self.req = req
        self.resp = resp
        self.doc = doc

@falcon.before(bunnies)
class WrappedClassResource:
    _some_fish = Fish()
    on_patch = {}

    @falcon.before(validate_param, 'limit')
    def on_get(self, req, resp, bunnies):
        if False:
            return 10
        self._capture(req, resp, bunnies)

    @falcon.before(validate_param, 'limit')
    def on_head(self, req, resp, bunnies):
        if False:
            print('Hello World!')
        self._capture(req, resp, bunnies)

    @falcon.before(_some_fish)
    def on_post(self, req, resp, fish, bunnies):
        if False:
            while True:
                i = 10
        self._capture(req, resp, bunnies)
        self.fish = fish

    @falcon.before(_some_fish.hook)
    def on_put(self, req, resp, fish, bunnies):
        if False:
            print('Hello World!')
        self._capture(req, resp, bunnies)
        self.fish = fish

    def _capture(self, req, resp, bunnies):
        if False:
            return 10
        self.req = req
        self.resp = resp
        self.bunnies = bunnies

@falcon.before(bunnies)
class ClassResourceWithAwareHooks:
    hook_as_class = ResourceAwareValidateParam()

    @falcon.before(validate_param, 'limit', 10)
    def on_get(self, req, resp, bunnies):
        if False:
            return 10
        self._capture(req, resp, bunnies)

    @falcon.before(validate_param, 'limit')
    def on_head(self, req, resp, bunnies):
        if False:
            for i in range(10):
                print('nop')
        self._capture(req, resp, bunnies)

    @falcon.before(hook_as_class)
    def on_put(self, req, resp, bunnies):
        if False:
            i = 10
            return i + 15
        self._capture(req, resp, bunnies)

    @falcon.before(hook_as_class.__call__)
    def on_post(self, req, resp, bunnies):
        if False:
            i = 10
            return i + 15
        self._capture(req, resp, bunnies)

    def _capture(self, req, resp, bunnies):
        if False:
            print('Hello World!')
        self.req = req
        self.resp = resp
        self.bunnies = bunnies

class TestFieldResource:

    @falcon.before(validate_field, field_name='id')
    def on_get(self, req, resp, id):
        if False:
            while True:
                i = 10
        self.id = id

class TestFieldResourceChild(TestFieldResource):

    def on_get(self, req, resp, id):
        if False:
            i = 10
            return i + 15
        super(TestFieldResourceChild, self).on_get(req, resp, id)

class TestFieldResourceChildToo(TestFieldResource):

    def on_get(self, req, resp, id):
        if False:
            print('Hello World!')
        super(TestFieldResourceChildToo, self).on_get(req, resp, id=id)

@falcon.before(bunnies)
@falcon.before(frogs)
@falcon.before(Fish())
@falcon.before(bunnies_in_the_head)
@falcon.before(frogs_in_the_head)
class ZooResource:

    def on_get(self, req, resp, bunnies, frogs, fish):
        if False:
            i = 10
            return i + 15
        self.bunnies = bunnies
        self.frogs = frogs
        self.fish = fish

class ZooResourceChild(ZooResource):

    def on_get(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        super(ZooResourceChild, self).on_get(req, resp, 'fluffy', 'not fluffy', fish='slippery')

@pytest.fixture
def wrapped_aware_resource():
    if False:
        print('Hello World!')
    return ClassResourceWithAwareHooks()

@pytest.fixture
def wrapped_resource():
    if False:
        return 10
    return WrappedClassResource()

@pytest.fixture
def resource():
    if False:
        print('Hello World!')
    return WrappedRespondersResource()

@pytest.fixture
def client(asgi, request, resource):
    if False:
        print('Hello World!')
    app = create_app(asgi)
    app.add_route('/', resource)
    return testing.TestClient(app)

@pytest.mark.parametrize('resource', [ZooResource(), ZooResourceChild()])
def test_multiple_resource_hooks(client, resource):
    if False:
        for i in range(10):
            print('nop')
    client.app.add_route('/', resource)
    result = client.simulate_get('/')
    assert 'not fluffy' == result.headers['X-Frogs']
    assert 'fluffy' == result.headers['X-Bunnies']
    assert 'fluffy' == resource.bunnies
    assert 'not fluffy' == resource.frogs
    assert 'slippery' == resource.fish

def test_input_validator(client):
    if False:
        while True:
            i = 10
    result = client.simulate_put('/')
    assert result.status_code == 400

def test_input_validator_inherited(client):
    if False:
        print('Hello World!')
    client.app.add_route('/', WrappedRespondersResourceChild())
    result = client.simulate_put('/')
    assert result.status_code == 400
    result = client.simulate_get('/', query_string='x=1000')
    assert result.status_code == 200
    result = client.simulate_get('/', query_string='x=1001')
    assert result.status_code == 400

def test_param_validator(client):
    if False:
        i = 10
        return i + 15
    result = client.simulate_get('/', query_string='limit=10', body='{}')
    assert result.status_code == 200
    result = client.simulate_get('/', query_string='limit=101')
    assert result.status_code == 400

@pytest.mark.parametrize('resource', [TestFieldResource(), TestFieldResourceChild(), TestFieldResourceChildToo()])
def test_field_validator(client, resource):
    if False:
        for i in range(10):
            print('nop')
    client.app.add_route('/queue/{id}/messages', resource)
    result = client.simulate_get('/queue/10/messages')
    assert result.status_code == 200
    assert resource.id == 10
    result = client.simulate_get('/queue/bogus/messages')
    assert result.status_code == 400

@pytest.mark.parametrize('body,doc', [(json.dumps({'animal': 'falcon'}), {'animal': 'falcon'}), ('{}', {}), ('', None), (None, None)])
def test_parser_sync(body, doc):
    if False:
        i = 10
        return i + 15
    app = falcon.App()
    resource = WrappedRespondersBodyParserResource()
    app.add_route('/', resource)
    testing.simulate_get(app, '/', body=body)
    assert resource.doc == doc

@pytest.mark.parametrize('body,doc', [(json.dumps({'animal': 'falcon'}), {'animal': 'falcon'}), ('{}', {}), ('', None), (None, None)])
def test_parser_async(body, doc):
    if False:
        print('Hello World!')
    with disable_asgi_non_coroutine_wrapping():

        class WrappedRespondersBodyParserAsyncResource:

            @falcon.before(validate_param_async, 'limit', 100, is_async=True)
            @falcon.before(parse_body_async)
            async def on_get(self, req, resp, doc=None):
                self.doc = doc

            @falcon.before(parse_body_async, is_async=False)
            async def on_put(self, req, resp, doc=None):
                self.doc = doc
    app = create_app(asgi=True)
    resource = WrappedRespondersBodyParserAsyncResource()
    app.add_route('/', resource)
    testing.simulate_get(app, '/', body=body)
    assert resource.doc == doc
    testing.simulate_put(app, '/', body=body)
    assert resource.doc == doc

    async def test_direct():
        resource = WrappedRespondersBodyParserAsyncResource()
        req = testing.create_asgi_req()
        resp = create_resp(True)
        await resource.on_get(req, resp, doc)
        assert resource.doc == doc
    falcon.async_to_sync(test_direct)

def test_wrapped_resource(client, wrapped_resource):
    if False:
        return 10
    client.app.add_route('/wrapped', wrapped_resource)
    result = client.simulate_patch('/wrapped')
    assert result.status_code == 405
    result = client.simulate_get('/wrapped', query_string='limit=10')
    assert result.status_code == 200
    assert 'fuzzy' == wrapped_resource.bunnies
    result = client.simulate_head('/wrapped')
    assert result.status_code == 200
    assert 'fuzzy' == wrapped_resource.bunnies
    result = client.simulate_post('/wrapped')
    assert result.status_code == 200
    assert 'slippery' == wrapped_resource.fish
    result = client.simulate_get('/wrapped', query_string='limit=101')
    assert result.status_code == 400
    assert wrapped_resource.bunnies == 'fuzzy'

def test_wrapped_resource_with_hooks_aware_of_resource(client, wrapped_aware_resource):
    if False:
        i = 10
        return i + 15
    client.app.add_route('/wrapped_aware', wrapped_aware_resource)
    result = client.simulate_patch('/wrapped_aware')
    assert result.status_code == 405
    result = client.simulate_get('/wrapped_aware', query_string='limit=10')
    assert result.status_code == 200
    assert wrapped_aware_resource.bunnies == 'fuzzy'
    for method in ('HEAD', 'PUT', 'POST'):
        result = client.simulate_request(method, '/wrapped_aware')
        assert result.status_code == 200
        assert wrapped_aware_resource.bunnies == 'fuzzy'
    result = client.simulate_get('/wrapped_aware', query_string='limit=11')
    assert result.status_code == 400
    assert wrapped_aware_resource.bunnies == 'fuzzy'
_another_fish = Fish()

def header_hook(req, resp, resource, params):
    if False:
        return 10
    value = resp.get_header('X-Hook-Applied') or '0'
    resp.set_header('X-Hook-Applied', str(int(value) + 1))

@falcon.before(header_hook)
class PiggybackingCollection:

    def __init__(self):
        if False:
            print('Hello World!')
        self._items = {}
        self._sequence = 0

    @falcon.before(_another_fish.hook)
    def on_delete(self, req, resp, fish, itemid):
        if False:
            i = 10
            return i + 15
        del self._items[itemid]
        resp.set_header('X-Fish-Trait', fish)
        resp.status = falcon.HTTP_NO_CONTENT

    @falcon.before(header_hook)
    @falcon.before(_another_fish.hook)
    @falcon.before(header_hook)
    def on_delete_collection(self, req, resp, fish):
        if False:
            return 10
        if fish != 'wet':
            raise falcon.HTTPUnavailableForLegalReasons(title='fish must be wet')
        self._items = {}
        resp.status = falcon.HTTP_NO_CONTENT

    @falcon.before(_another_fish)
    def on_get(self, req, resp, fish, itemid):
        if False:
            while True:
                i = 10
        resp.set_header('X-Fish-Trait', fish)
        resp.media = self._items[itemid]

    def on_get_collection(self, req, resp):
        if False:
            while True:
                i = 10
        resp.media = sorted(self._items.values(), key=lambda item: item['itemid'])

    def on_head_(self):
        if False:
            while True:
                i = 10
        return 'I shall not be decorated.'

    def on_header(self):
        if False:
            i = 10
            return i + 15
        return 'I shall not be decorated.'

    def on_post_collection(self, req, resp):
        if False:
            return 10
        self._sequence += 1
        itemid = self._sequence
        self._items[itemid] = dict(req.media, itemid=itemid)
        resp.location = '/items/{}'.format(itemid)
        resp.status = falcon.HTTP_CREATED

class PiggybackingCollectionAsync(PiggybackingCollection):

    @falcon.before(header_hook)
    async def on_post_collection(self, req, resp):
        self._sequence += 1
        itemid = self._sequence
        doc = await req.get_media()
        self._items[itemid] = dict(doc, itemid=itemid)
        resp.location = '/items/{}'.format(itemid)
        resp.status = falcon.HTTP_CREATED

@pytest.fixture(params=[True, False])
def app_client(request):
    if False:
        for i in range(10):
            print('nop')
    items = PiggybackingCollectionAsync() if request.param else PiggybackingCollection()
    app = create_app(asgi=request.param)
    app.add_route('/items', items, suffix='collection')
    app.add_route('/items/{itemid:int}', items)
    return testing.TestClient(app)

def test_piggybacking_resource_post_item(app_client):
    if False:
        for i in range(10):
            print('nop')
    resp1 = app_client.simulate_post('/items', json={'color': 'green'})
    assert resp1.status_code == 201
    assert 'X-Fish-Trait' not in resp1.headers
    assert resp1.headers['Location'] == '/items/1'
    assert resp1.headers['X-Hook-Applied'] == '1'
    resp2 = app_client.simulate_get(resp1.headers['Location'])
    assert resp2.status_code == 200
    assert resp2.headers['X-Fish-Trait'] == 'slippery'
    assert resp2.headers['X-Hook-Applied'] == '1'
    assert resp2.json == {'color': 'green', 'itemid': 1}
    resp3 = app_client.simulate_get('/items')
    assert resp3.status_code == 200
    assert 'X-Fish-Trait' not in resp3.headers
    assert resp3.headers['X-Hook-Applied'] == '1'
    assert resp3.json == [{'color': 'green', 'itemid': 1}]

def test_piggybacking_resource_post_and_delete(app_client):
    if False:
        while True:
            i = 10
    for number in range(1, 8):
        resp = app_client.simulate_post('/items', json={'number': number})
        assert resp.status_code == 201
        assert resp.headers['X-Hook-Applied'] == '1'
        assert len(app_client.simulate_get('/items').json) == number
    resp = app_client.simulate_delete('/items/{}'.format(number))
    assert resp.status_code == 204
    assert resp.headers['X-Fish-Trait'] == 'wet'
    assert resp.headers['X-Hook-Applied'] == '1'
    assert len(app_client.simulate_get('/items').json) == 6
    resp = app_client.simulate_delete('/items')
    assert resp.status_code == 204
    assert resp.headers['X-Hook-Applied'] == '3'
    assert app_client.simulate_get('/items').json == []

def test_decorable_name_pattern():
    if False:
        for i in range(10):
            print('nop')
    resource = PiggybackingCollection()
    assert resource.on_head_() == 'I shall not be decorated.'
    assert resource.on_header() == 'I shall not be decorated.'
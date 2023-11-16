import pytest
import falcon
from falcon import testing
from falcon.util.deprecation import AttributeRemovedError
from _util import create_app, create_resp

@pytest.fixture
def resp(asgi):
    if False:
        i = 10
        return i + 15
    return create_resp(asgi)

def test_append_body(resp):
    if False:
        i = 10
        return i + 15
    text = 'Hello beautiful world! '
    resp.text = ''
    with pytest.raises(AttributeRemovedError):
        resp.body = 'x'
    for token in text.split():
        resp.text += token
        resp.text += ' '
    assert resp.text == text
    for ErrorType in (AttributeError, AttributeRemovedError):
        with pytest.raises(ErrorType):
            resp.body

def test_response_repr(resp):
    if False:
        print('Hello World!')
    _repr = '<%s: %s>' % (resp.__class__.__name__, resp.status)
    assert resp.__repr__() == _repr

def test_content_length_set_on_head_with_no_body(asgi):
    if False:
        while True:
            i = 10

    class NoBody:

        def on_get(self, req, resp):
            if False:
                for i in range(10):
                    print('nop')
            pass
        on_head = on_get
    app = create_app(asgi)
    app.add_route('/', NoBody())
    result = testing.simulate_head(app, '/')
    assert result.status_code == 200
    assert result.headers['content-length'] == '0'

@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_content_length_not_set_when_streaming_response(asgi, method):
    if False:
        for i in range(10):
            print('nop')

    class SynthesizedHead:

        def on_get(self, req, resp):
            if False:
                print('Hello World!')

            def words():
                if False:
                    print('Hello World!')
                for word in ('Hello', ',', ' ', 'World!'):
                    yield word.encode()
            resp.content_type = falcon.MEDIA_TEXT
            resp.stream = words()
        on_head = on_get

    class SynthesizedHeadAsync:

        async def on_get(self, req, resp):

            class Words:

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    self._stream = iter(('Hello', ',', ' ', 'World!'))

                def __aiter__(self):
                    if False:
                        return 10
                    return self

                async def __anext__(self):
                    try:
                        return next(self._stream).encode()
                    except StopIteration:
                        pass
            resp.content_type = falcon.MEDIA_TEXT
            resp.stream = Words()
        on_head = on_get
    app = create_app(asgi)
    app.add_route('/', SynthesizedHeadAsync() if asgi else SynthesizedHead())
    result = testing.simulate_request(app, method)
    assert result.status_code == 200
    assert result.headers['content-type'] == falcon.MEDIA_TEXT
    assert 'content-length' not in result.headers
    if method == 'GET':
        assert result.text == 'Hello, World!'

class CodeResource:

    def on_get(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        resp.content_type = 'text/x-malbolge'
        resp.media = '\'&%$#"!76543210/43,P0).\'&%I6'
        resp.status = falcon.HTTP_725

def test_unsupported_response_content_type(asgi):
    if False:
        i = 10
        return i + 15
    app = create_app(asgi)
    app.add_route('/test.mal', CodeResource())
    resp = testing.simulate_get(app, '/test.mal')
    assert resp.status_code == 415

def test_response_body_rendition_error(asgi):
    if False:
        for i in range(10):
            print('nop')

    class MalbolgeHandler(falcon.media.BaseHandler):

        def serialize(self, media, content_type):
            if False:
                i = 10
                return i + 15
            raise falcon.HTTPError(falcon.HTTP_753)
    app = create_app(asgi)
    app.resp_options.media_handlers['text/x-malbolge'] = MalbolgeHandler()
    app.add_route('/test.mal', CodeResource())
    resp = testing.simulate_get(app, '/test.mal')
    assert resp.status_code == 753
    del app.resp_options.media_handlers['text/x-malbolge']
    resp = testing.simulate_get(app, '/test.mal')
    assert resp.status_code == 415
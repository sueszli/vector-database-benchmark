import logging
import pytest
import sanic
from sanic import Sanic
from sanic.config import Config
from sanic.errorpages import TextRenderer, exception_response, guess_mime
from sanic.exceptions import NotFound, SanicException
from sanic.handlers import ErrorHandler
from sanic.request import Request
from sanic.response import HTTPResponse, empty, html, json, text

@pytest.fixture
def app():
    if False:
        i = 10
        return i + 15
    app = Sanic('error_page_testing')

    @app.route('/error', methods=['GET', 'POST'])
    def err(request):
        if False:
            while True:
                i = 10
        raise Exception('something went wrong')

    @app.get('/forced_json/<fail>', error_format='json')
    def manual_fail(request, fail):
        if False:
            while True:
                i = 10
        if fail == 'fail':
            raise Exception
        return html('')

    @app.get('/empty/<fail>')
    def empty_fail(request, fail):
        if False:
            return 10
        if fail == 'fail':
            raise Exception
        return empty()

    @app.get('/json/<fail>')
    def json_fail(request, fail):
        if False:
            return 10
        if fail == 'fail':
            raise Exception
        return json({'foo': 'bar'}) if fail == 'json' else empty()

    @app.get('/html/<fail>')
    def html_fail(request, fail):
        if False:
            i = 10
            return i + 15
        if fail == 'fail':
            raise Exception
        return html('<h1>foo</h1>')

    @app.get('/text/<fail>')
    def text_fail(request, fail):
        if False:
            return 10
        if fail == 'fail':
            raise Exception
        return text('foo')

    @app.get('/mixed/<param>')
    def mixed_fail(request, param):
        if False:
            i = 10
            return i + 15
        if param not in ('json', 'html'):
            raise Exception
        return json({}) if param == 'json' else html('')
    return app

@pytest.fixture
def fake_request(app):
    if False:
        print('Hello World!')
    return Request(b'/foobar', {'accept': '*/*'}, '1.1', 'GET', None, app)

@pytest.mark.parametrize('fallback,content_type, exception, status', ((None, 'text/plain; charset=utf-8', Exception, 500), ('html', 'text/html; charset=utf-8', Exception, 500), ('auto', 'text/plain; charset=utf-8', Exception, 500), ('text', 'text/plain; charset=utf-8', Exception, 500), ('json', 'application/json', Exception, 500), (None, 'text/plain; charset=utf-8', NotFound, 404), ('html', 'text/html; charset=utf-8', NotFound, 404), ('auto', 'text/plain; charset=utf-8', NotFound, 404), ('text', 'text/plain; charset=utf-8', NotFound, 404), ('json', 'application/json', NotFound, 404)))
def test_should_return_html_valid_setting(fake_request, fallback, content_type, exception, status):
    if False:
        while True:
            i = 10
    if fallback:
        fake_request.app.config.FALLBACK_ERROR_FORMAT = fallback
    try:
        raise exception('bad stuff')
    except Exception as e:
        response = exception_response(fake_request, e, True, base=TextRenderer, fallback=fake_request.app.config.FALLBACK_ERROR_FORMAT)
    assert isinstance(response, HTTPResponse)
    assert response.status == status
    assert response.content_type == content_type

def test_auto_fallback_with_data(app):
    if False:
        return 10
    app.config.FALLBACK_ERROR_FORMAT = 'auto'
    (_, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.post('/error', json={'foo': 'bar'})
    assert response.status == 500
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.post('/error', data={'foo': 'bar'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_auto_fallback_with_content_type(app):
    if False:
        print('Hello World!')
    app.config.FALLBACK_ERROR_FORMAT = 'auto'
    (_, response) = app.test_client.get('/error', headers={'content-type': 'application/json', 'accept': '*/*'})
    assert response.status == 500
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.get('/error', headers={'content-type': 'foo/bar', 'accept': '*/*'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_route_error_format_set_on_auto(app):
    if False:
        return 10

    @app.get('/text')
    def text_response(request):
        if False:
            i = 10
            return i + 15
        return text(request.route.extra.error_format)

    @app.get('/json')
    def json_response(request):
        if False:
            print('Hello World!')
        return json({'format': request.route.extra.error_format})

    @app.get('/html')
    def html_response(request):
        if False:
            return 10
        return html(request.route.extra.error_format)
    (_, response) = app.test_client.get('/text')
    assert response.text == 'text'
    (_, response) = app.test_client.get('/json')
    assert response.json['format'] == 'json'
    (_, response) = app.test_client.get('/html')
    assert response.text == 'html'

def test_route_error_response_from_auto_route(app):
    if False:
        print('Hello World!')

    @app.get('/text')
    def text_response(request):
        if False:
            return 10
        raise Exception('oops')
        return text('Never gonna see this')

    @app.get('/json')
    def json_response(request):
        if False:
            i = 10
            return i + 15
        raise Exception('oops')
        return json({'message': 'Never gonna see this'})

    @app.get('/html')
    def html_response(request):
        if False:
            return 10
        raise Exception('oops')
        return html('<h1>Never gonna see this</h1>')
    (_, response) = app.test_client.get('/text')
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/json')
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.get('/html')
    assert response.content_type == 'text/html; charset=utf-8'

def test_route_error_response_from_explicit_format(app):
    if False:
        i = 10
        return i + 15

    @app.get('/text', error_format='json')
    def text_response(request):
        if False:
            print('Hello World!')
        raise Exception('oops')
        return text('Never gonna see this')

    @app.get('/json', error_format='text')
    def json_response(request):
        if False:
            i = 10
            return i + 15
        raise Exception('oops')
        return json({'message': 'Never gonna see this'})
    (_, response) = app.test_client.get('/text')
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.get('/json')
    assert response.content_type == 'text/plain; charset=utf-8'

def test_blueprint_error_response_from_explicit_format(app):
    if False:
        for i in range(10):
            print('nop')
    bp = sanic.Blueprint('MyBlueprint')

    @bp.get('/text', error_format='json')
    def text_response(request):
        if False:
            return 10
        raise Exception('oops')
        return text('Never gonna see this')

    @bp.get('/json', error_format='text')
    def json_response(request):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('oops')
        return json({'message': 'Never gonna see this'})
    app.blueprint(bp)
    (_, response) = app.test_client.get('/text')
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.get('/json')
    assert response.content_type == 'text/plain; charset=utf-8'

def test_unknown_fallback_format(app):
    if False:
        print('Hello World!')
    with pytest.raises(SanicException, match='Unknown format: bad'):
        app.config.FALLBACK_ERROR_FORMAT = 'bad'

def test_route_error_format_unknown(app):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(SanicException, match='Unknown format: bad'):

        @app.get('/text', error_format='bad')
        def handler(request):
            if False:
                while True:
                    i = 10
            ...

def test_fallback_with_content_type_html(app):
    if False:
        print('Hello World!')
    app.config.FALLBACK_ERROR_FORMAT = 'auto'
    (_, response) = app.test_client.get('/error', headers={'content-type': 'application/json', 'accept': 'text/html'})
    assert response.status == 500
    assert response.content_type == 'text/html; charset=utf-8'

def test_fallback_with_content_type_mismatch_accept(app):
    if False:
        print('Hello World!')
    app.config.FALLBACK_ERROR_FORMAT = 'auto'
    (_, response) = app.test_client.get('/error', headers={'content-type': 'application/json', 'accept': 'text/plain'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/error', headers={'content-type': 'text/html', 'accept': 'foo/bar'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    app.router.reset()

    @app.route('/alt1', name='alt1')
    @app.route('/alt2', error_format='text', name='alt2')
    @app.route('/alt3', error_format='html', name='alt3')
    def handler(_):
        if False:
            i = 10
            return i + 15
        raise Exception('problem here')
        return json({})
    app.router.finalize()
    (_, response) = app.test_client.get('/alt1', headers={'accept': 'foo/bar'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/alt1', headers={'accept': 'foo/bar,*/*'})
    assert response.status == 500
    assert response.content_type == 'application/json'
    (_, response) = app.test_client.get('/alt2', headers={'accept': 'foo/bar'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/alt2', headers={'accept': 'foo/bar,*/*'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/alt3', headers={'accept': 'foo/bar'})
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'
    (_, response) = app.test_client.get('/alt3', headers={'accept': 'foo/bar,text/html'})
    assert response.status == 500
    assert response.content_type == 'text/html; charset=utf-8'

@pytest.mark.parametrize('accept,content_type,expected', ((None, None, 'text/plain; charset=utf-8'), ('foo/bar', None, 'text/plain; charset=utf-8'), ('application/json', None, 'application/json'), ('application/json,text/plain', None, 'application/json'), ('text/plain,application/json', None, 'application/json'), ('text/plain,foo/bar', None, 'text/plain; charset=utf-8'), ('text/plain,text/html', None, 'text/plain; charset=utf-8'), ('*/*', 'foo/bar', 'text/plain; charset=utf-8'), ('*/*', 'application/json', 'application/json'), ('text/*,*/plain', None, 'text/plain; charset=utf-8')))
def test_combinations_for_auto(fake_request, accept, content_type, expected):
    if False:
        return 10
    if accept:
        fake_request.headers['accept'] = accept
    else:
        del fake_request.headers['accept']
    if content_type:
        fake_request.headers['content-type'] = content_type
    try:
        raise Exception('bad stuff')
    except Exception as e:
        response = exception_response(fake_request, e, True, base=TextRenderer, fallback='auto')
    assert response.content_type == expected

def test_allow_fallback_error_format_set_main_process_start(app):
    if False:
        i = 10
        return i + 15

    @app.main_process_start
    async def start(app, _):
        app.config.FALLBACK_ERROR_FORMAT = 'text'
    (_, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_setting_fallback_on_config_changes_as_expected(app):
    if False:
        for i in range(10):
            print('nop')
    app.error_handler = ErrorHandler()
    (_, response) = app.test_client.get('/error')
    assert response.content_type == 'text/plain; charset=utf-8'
    app.config.FALLBACK_ERROR_FORMAT = 'html'
    (_, response) = app.test_client.get('/error')
    assert response.content_type == 'text/html; charset=utf-8'
    app.config.FALLBACK_ERROR_FORMAT = 'text'
    (_, response) = app.test_client.get('/error')
    assert response.content_type == 'text/plain; charset=utf-8'

def test_allow_fallback_error_format_in_config_injection():
    if False:
        return 10

    class MyConfig(Config):
        FALLBACK_ERROR_FORMAT = 'text'
    app = Sanic('test', config=MyConfig())

    @app.route('/error', methods=['GET', 'POST'])
    def err(request):
        if False:
            while True:
                i = 10
        raise Exception('something went wrong')
    (request, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_allow_fallback_error_format_in_config_replacement(app):
    if False:
        return 10

    class MyConfig(Config):
        FALLBACK_ERROR_FORMAT = 'text'
    app.config = MyConfig()
    (request, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_config_fallback_before_and_after_startup(app):
    if False:
        i = 10
        return i + 15
    app.config.FALLBACK_ERROR_FORMAT = 'json'

    @app.main_process_start
    async def start(app, _):
        app.config.FALLBACK_ERROR_FORMAT = 'text'
    (_, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'application/json'

def test_config_fallback_using_update_dict(app):
    if False:
        for i in range(10):
            print('nop')
    app.config.update({'FALLBACK_ERROR_FORMAT': 'text'})
    (_, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_config_fallback_using_update_kwarg(app):
    if False:
        while True:
            i = 10
    app.config.update(FALLBACK_ERROR_FORMAT='text')
    (_, response) = app.test_client.get('/error')
    assert response.status == 500
    assert response.content_type == 'text/plain; charset=utf-8'

def test_config_fallback_bad_value(app):
    if False:
        print('Hello World!')
    message = 'Unknown format: fake'
    with pytest.raises(SanicException, match=message):
        app.config.FALLBACK_ERROR_FORMAT = 'fake'

@pytest.mark.parametrize('route_format,fallback,accept,expected', (('json', 'html', '*/*', "The client accepts */*, using 'json' from fakeroute"), ('json', 'auto', 'text/html,*/*;q=0.8', "The client accepts text/html, using 'html' from any"), ('json', 'json', 'text/html,*/*;q=0.8', "The client accepts */*;q=0.8, using 'json' from fakeroute"), ('', 'html', 'text/*,*/plain', "The client accepts text/*, using 'html' from FALLBACK_ERROR_FORMAT"), ('', 'json', 'text/*,*/*', "The client accepts */*, using 'json' from FALLBACK_ERROR_FORMAT"), ('', 'auto', '*/*,application/json;q=0.5', "The client accepts */*, using 'json' from request.accept"), ('', 'auto', '*/*', "The client accepts */*, using 'json' from content-type"), ('', 'auto', 'text/html,text/plain', "The client accepts text/plain, using 'text' from any"), ('', 'auto', 'text/html,text/plain;q=0.9', "The client accepts text/html, using 'html' from any"), ('html', 'json', 'application/xml', 'No format found, the client accepts [application/xml]'), ('', 'auto', '*/*', "The client accepts */*, using 'text' from any"), ('', '', '*/*', 'No format found, the client accepts [*/*]'), ('', 'auto', '*/*', "The client accepts */*, using 'json' from request.json")))
def test_guess_mime_logging(caplog, fake_request, route_format, fallback, accept, expected):
    if False:
        while True:
            i = 10

    class FakeObject:
        pass
    fake_request.route = FakeObject()
    fake_request.route.name = 'fakeroute'
    fake_request.route.extra = FakeObject()
    fake_request.route.extra.error_format = route_format
    if accept is None:
        del fake_request.headers['accept']
    else:
        fake_request.headers['accept'] = accept
    if 'content-type' in expected:
        fake_request.headers['content-type'] = 'application/json'
    if 'request.json' in expected:
        fake_request.parsed_json = {'foo': 'bar'}
    with caplog.at_level(logging.DEBUG, logger='sanic.root'):
        guess_mime(fake_request, fallback)
    (logmsg,) = [r.message for r in caplog.records if r.funcName == 'guess_mime']
    assert logmsg == expected

@pytest.mark.parametrize('format,expected', (('html', 'text/html; charset=utf-8'), ('text', 'text/plain; charset=utf-8'), ('json', 'application/json')))
def test_exception_header_on_renderers(app: Sanic, format, expected):
    if False:
        i = 10
        return i + 15
    app.config.FALLBACK_ERROR_FORMAT = format

    @app.get('/test')
    def test(request):
        if False:
            while True:
                i = 10
        raise SanicException('test', status_code=400, headers={'exception': 'test'})
    (_, response) = app.test_client.get('/test')
    assert response.status == 400
    assert response.headers.get('exception') == 'test'
    assert response.content_type == expected
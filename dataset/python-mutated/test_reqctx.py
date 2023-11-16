import warnings
import pytest
import flask
from flask.globals import request_ctx
from flask.sessions import SecureCookieSessionInterface
from flask.sessions import SessionInterface
try:
    from greenlet import greenlet
except ImportError:
    greenlet = None

def test_teardown_on_pop(app):
    if False:
        return 10
    buffer = []

    @app.teardown_request
    def end_of_request(exception):
        if False:
            i = 10
            return i + 15
        buffer.append(exception)
    ctx = app.test_request_context()
    ctx.push()
    assert buffer == []
    ctx.pop()
    assert buffer == [None]

def test_teardown_with_previous_exception(app):
    if False:
        for i in range(10):
            print('nop')
    buffer = []

    @app.teardown_request
    def end_of_request(exception):
        if False:
            print('Hello World!')
        buffer.append(exception)
    try:
        raise Exception('dummy')
    except Exception:
        pass
    with app.test_request_context():
        assert buffer == []
    assert buffer == [None]

def test_teardown_with_handled_exception(app):
    if False:
        print('Hello World!')
    buffer = []

    @app.teardown_request
    def end_of_request(exception):
        if False:
            print('Hello World!')
        buffer.append(exception)
    with app.test_request_context():
        assert buffer == []
        try:
            raise Exception('dummy')
        except Exception:
            pass
    assert buffer == [None]

def test_proper_test_request_context(app):
    if False:
        print('Hello World!')
    app.config.update(SERVER_NAME='localhost.localdomain:5000')

    @app.route('/')
    def index():
        if False:
            print('Hello World!')
        return None

    @app.route('/', subdomain='foo')
    def sub():
        if False:
            return 10
        return None
    with app.test_request_context('/'):
        assert flask.url_for('index', _external=True) == 'http://localhost.localdomain:5000/'
    with app.test_request_context('/'):
        assert flask.url_for('sub', _external=True) == 'http://foo.localhost.localdomain:5000/'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Current server name', UserWarning, 'flask.app')
        with app.test_request_context('/', environ_overrides={'HTTP_HOST': 'localhost'}):
            pass
    app.config.update(SERVER_NAME='localhost')
    with app.test_request_context('/', environ_overrides={'SERVER_NAME': 'localhost'}):
        pass
    app.config.update(SERVER_NAME='localhost:80')
    with app.test_request_context('/', environ_overrides={'SERVER_NAME': 'localhost:80'}):
        pass

def test_context_binding(app):
    if False:
        print('Hello World!')

    @app.route('/')
    def index():
        if False:
            print('Hello World!')
        return f"Hello {flask.request.args['name']}!"

    @app.route('/meh')
    def meh():
        if False:
            return 10
        return flask.request.url
    with app.test_request_context('/?name=World'):
        assert index() == 'Hello World!'
    with app.test_request_context('/meh'):
        assert meh() == 'http://localhost/meh'
    assert not flask.request

def test_context_test(app):
    if False:
        for i in range(10):
            print('nop')
    assert not flask.request
    assert not flask.has_request_context()
    ctx = app.test_request_context()
    ctx.push()
    try:
        assert flask.request
        assert flask.has_request_context()
    finally:
        ctx.pop()

def test_manual_context_binding(app):
    if False:
        print('Hello World!')

    @app.route('/')
    def index():
        if False:
            i = 10
            return i + 15
        return f"Hello {flask.request.args['name']}!"
    ctx = app.test_request_context('/?name=World')
    ctx.push()
    assert index() == 'Hello World!'
    ctx.pop()
    with pytest.raises(RuntimeError):
        index()

@pytest.mark.skipif(greenlet is None, reason='greenlet not installed')
class TestGreenletContextCopying:

    def test_greenlet_context_copying(self, app, client):
        if False:
            return 10
        greenlets = []

        @app.route('/')
        def index():
            if False:
                return 10
            flask.session['fizz'] = 'buzz'
            reqctx = request_ctx.copy()

            def g():
                if False:
                    for i in range(10):
                        print('nop')
                assert not flask.request
                assert not flask.current_app
                with reqctx:
                    assert flask.request
                    assert flask.current_app == app
                    assert flask.request.path == '/'
                    assert flask.request.args['foo'] == 'bar'
                    assert flask.session.get('fizz') == 'buzz'
                assert not flask.request
                return 42
            greenlets.append(greenlet(g))
            return 'Hello World!'
        rv = client.get('/?foo=bar')
        assert rv.data == b'Hello World!'
        result = greenlets[0].run()
        assert result == 42

    def test_greenlet_context_copying_api(self, app, client):
        if False:
            i = 10
            return i + 15
        greenlets = []

        @app.route('/')
        def index():
            if False:
                print('Hello World!')
            flask.session['fizz'] = 'buzz'

            @flask.copy_current_request_context
            def g():
                if False:
                    print('Hello World!')
                assert flask.request
                assert flask.current_app == app
                assert flask.request.path == '/'
                assert flask.request.args['foo'] == 'bar'
                assert flask.session.get('fizz') == 'buzz'
                return 42
            greenlets.append(greenlet(g))
            return 'Hello World!'
        rv = client.get('/?foo=bar')
        assert rv.data == b'Hello World!'
        result = greenlets[0].run()
        assert result == 42

def test_session_error_pops_context():
    if False:
        for i in range(10):
            print('nop')

    class SessionError(Exception):
        pass

    class FailingSessionInterface(SessionInterface):

        def open_session(self, app, request):
            if False:
                print('Hello World!')
            raise SessionError()

    class CustomFlask(flask.Flask):
        session_interface = FailingSessionInterface()
    app = CustomFlask(__name__)

    @app.route('/')
    def index():
        if False:
            for i in range(10):
                print('nop')
        AssertionError()
    response = app.test_client().get('/')
    assert response.status_code == 500
    assert not flask.request
    assert not flask.current_app

def test_session_dynamic_cookie_name():
    if False:
        print('Hello World!')

    class PathAwareSessionInterface(SecureCookieSessionInterface):

        def get_cookie_name(self, app):
            if False:
                i = 10
                return i + 15
            if flask.request.url.endswith('dynamic_cookie'):
                return 'dynamic_cookie_name'
            else:
                return super().get_cookie_name(app)

    class CustomFlask(flask.Flask):
        session_interface = PathAwareSessionInterface()
    app = CustomFlask(__name__)
    app.secret_key = 'secret_key'

    @app.route('/set', methods=['POST'])
    def set():
        if False:
            i = 10
            return i + 15
        flask.session['value'] = flask.request.form['value']
        return 'value set'

    @app.route('/get')
    def get():
        if False:
            print('Hello World!')
        v = flask.session.get('value', 'None')
        return v

    @app.route('/set_dynamic_cookie', methods=['POST'])
    def set_dynamic_cookie():
        if False:
            i = 10
            return i + 15
        flask.session['value'] = flask.request.form['value']
        return 'value set'

    @app.route('/get_dynamic_cookie')
    def get_dynamic_cookie():
        if False:
            for i in range(10):
                print('nop')
        v = flask.session.get('value', 'None')
        return v
    test_client = app.test_client()
    assert test_client.post('/set', data={'value': '42'}).data == b'value set'
    assert test_client.post('/set_dynamic_cookie', data={'value': '616'}).data == b'value set'
    assert test_client.get('/get').data == b'42'
    assert test_client.get('/get_dynamic_cookie').data == b'616'

def test_bad_environ_raises_bad_request():
    if False:
        return 10
    app = flask.Flask(__name__)
    from flask.testing import EnvironBuilder
    builder = EnvironBuilder(app)
    environ = builder.get_environ()
    environ['HTTP_HOST'] = '\x8a'
    with app.request_context(environ):
        response = app.full_dispatch_request()
    assert response.status_code == 400

def test_environ_for_valid_idna_completes():
    if False:
        for i in range(10):
            print('nop')
    app = flask.Flask(__name__)

    @app.route('/')
    def index():
        if False:
            for i in range(10):
                print('nop')
        return 'Hello World!'
    from flask.testing import EnvironBuilder
    builder = EnvironBuilder(app)
    environ = builder.get_environ()
    environ['HTTP_HOST'] = 'ąśźäüжŠßя.com'
    with app.request_context(environ):
        response = app.full_dispatch_request()
    assert response.status_code == 200

def test_normal_environ_completes():
    if False:
        for i in range(10):
            print('nop')
    app = flask.Flask(__name__)

    @app.route('/')
    def index():
        if False:
            print('Hello World!')
        return 'Hello World!'
    response = app.test_client().get('/', headers={'host': 'xn--on-0ia.com'})
    assert response.status_code == 200
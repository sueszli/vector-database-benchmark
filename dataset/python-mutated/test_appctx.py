import pytest
import flask
from flask.globals import app_ctx
from flask.globals import request_ctx

def test_basic_url_generation(app):
    if False:
        for i in range(10):
            print('nop')
    app.config['SERVER_NAME'] = 'localhost'
    app.config['PREFERRED_URL_SCHEME'] = 'https'

    @app.route('/')
    def index():
        if False:
            print('Hello World!')
        pass
    with app.app_context():
        rv = flask.url_for('index')
        assert rv == 'https://localhost/'

def test_url_generation_requires_server_name(app):
    if False:
        return 10
    with app.app_context():
        with pytest.raises(RuntimeError):
            flask.url_for('index')

def test_url_generation_without_context_fails():
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError):
        flask.url_for('index')

def test_request_context_means_app_context(app):
    if False:
        i = 10
        return i + 15
    with app.test_request_context():
        assert flask.current_app._get_current_object() is app
    assert not flask.current_app

def test_app_context_provides_current_app(app):
    if False:
        print('Hello World!')
    with app.app_context():
        assert flask.current_app._get_current_object() is app
    assert not flask.current_app

def test_app_tearing_down(app):
    if False:
        print('Hello World!')
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        if False:
            print('Hello World!')
        cleanup_stuff.append(exception)
    with app.app_context():
        pass
    assert cleanup_stuff == [None]

def test_app_tearing_down_with_previous_exception(app):
    if False:
        return 10
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        if False:
            for i in range(10):
                print('nop')
        cleanup_stuff.append(exception)
    try:
        raise Exception('dummy')
    except Exception:
        pass
    with app.app_context():
        pass
    assert cleanup_stuff == [None]

def test_app_tearing_down_with_handled_exception_by_except_block(app):
    if False:
        i = 10
        return i + 15
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        if False:
            return 10
        cleanup_stuff.append(exception)
    with app.app_context():
        try:
            raise Exception('dummy')
        except Exception:
            pass
    assert cleanup_stuff == [None]

def test_app_tearing_down_with_handled_exception_by_app_handler(app, client):
    if False:
        while True:
            i = 10
    app.config['PROPAGATE_EXCEPTIONS'] = True
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        if False:
            while True:
                i = 10
        cleanup_stuff.append(exception)

    @app.route('/')
    def index():
        if False:
            for i in range(10):
                print('nop')
        raise Exception('dummy')

    @app.errorhandler(Exception)
    def handler(f):
        if False:
            i = 10
            return i + 15
        return flask.jsonify(str(f))
    with app.app_context():
        client.get('/')
    assert cleanup_stuff == [None]

def test_app_tearing_down_with_unhandled_exception(app, client):
    if False:
        i = 10
        return i + 15
    app.config['PROPAGATE_EXCEPTIONS'] = True
    cleanup_stuff = []

    @app.teardown_appcontext
    def cleanup(exception):
        if False:
            return 10
        cleanup_stuff.append(exception)

    @app.route('/')
    def index():
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('dummy')
    with pytest.raises(ValueError, match='dummy'):
        with app.app_context():
            client.get('/')
    assert len(cleanup_stuff) == 1
    assert isinstance(cleanup_stuff[0], ValueError)
    assert str(cleanup_stuff[0]) == 'dummy'

def test_app_ctx_globals_methods(app, app_ctx):
    if False:
        i = 10
        return i + 15
    assert flask.g.get('foo') is None
    assert flask.g.get('foo', 'bar') == 'bar'
    assert 'foo' not in flask.g
    flask.g.foo = 'bar'
    assert 'foo' in flask.g
    flask.g.setdefault('bar', 'the cake is a lie')
    flask.g.setdefault('bar', 'hello world')
    assert flask.g.bar == 'the cake is a lie'
    assert flask.g.pop('bar') == 'the cake is a lie'
    with pytest.raises(KeyError):
        flask.g.pop('bar')
    assert flask.g.pop('bar', 'more cake') == 'more cake'
    assert list(flask.g) == ['foo']
    assert repr(flask.g) == "<flask.g of 'flask_test'>"

def test_custom_app_ctx_globals_class(app):
    if False:
        while True:
            i = 10

    class CustomRequestGlobals:

        def __init__(self):
            if False:
                return 10
            self.spam = 'eggs'
    app.app_ctx_globals_class = CustomRequestGlobals
    with app.app_context():
        assert flask.render_template_string('{{ g.spam }}') == 'eggs'

def test_context_refcounts(app, client):
    if False:
        print('Hello World!')
    called = []

    @app.teardown_request
    def teardown_req(error=None):
        if False:
            while True:
                i = 10
        called.append('request')

    @app.teardown_appcontext
    def teardown_app(error=None):
        if False:
            for i in range(10):
                print('nop')
        called.append('app')

    @app.route('/')
    def index():
        if False:
            return 10
        with app_ctx:
            with request_ctx:
                pass
        assert flask.request.environ['werkzeug.request'] is not None
        return ''
    res = client.get('/')
    assert res.status_code == 200
    assert res.data == b''
    assert called == ['request', 'app']

def test_clean_pop(app):
    if False:
        return 10
    app.testing = False
    called = []

    @app.teardown_request
    def teardown_req(error=None):
        if False:
            for i in range(10):
                print('nop')
        raise ZeroDivisionError

    @app.teardown_appcontext
    def teardown_app(error=None):
        if False:
            for i in range(10):
                print('nop')
        called.append('TEARDOWN')
    with app.app_context():
        called.append(flask.current_app.name)
    assert called == ['flask_test', 'TEARDOWN']
    assert not flask.current_app
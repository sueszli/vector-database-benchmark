import pytest
from werkzeug.exceptions import Forbidden
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import InternalServerError
from werkzeug.exceptions import NotFound
import flask

def test_error_handler_no_match(app, client):
    if False:
        print('Hello World!')

    class CustomException(Exception):
        pass

    @app.errorhandler(CustomException)
    def custom_exception_handler(e):
        if False:
            i = 10
            return i + 15
        assert isinstance(e, CustomException)
        return 'custom'
    with pytest.raises(TypeError) as exc_info:
        app.register_error_handler(CustomException(), None)
    assert 'CustomException() is an instance, not a class.' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        app.register_error_handler(list, None)
    assert "'list' is not a subclass of Exception." in str(exc_info.value)

    @app.errorhandler(500)
    def handle_500(e):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(e, InternalServerError)
        if e.original_exception is not None:
            return f'wrapped {type(e.original_exception).__name__}'
        return 'direct'
    with pytest.raises(ValueError) as exc_info:
        app.register_error_handler(999, None)
    assert 'Use a subclass of HTTPException' in str(exc_info.value)

    @app.route('/custom')
    def custom_test():
        if False:
            for i in range(10):
                print('nop')
        raise CustomException()

    @app.route('/keyerror')
    def key_error():
        if False:
            i = 10
            return i + 15
        raise KeyError()

    @app.route('/abort')
    def do_abort():
        if False:
            i = 10
            return i + 15
        flask.abort(500)
    app.testing = False
    assert client.get('/custom').data == b'custom'
    assert client.get('/keyerror').data == b'wrapped KeyError'
    assert client.get('/abort').data == b'direct'

def test_error_handler_subclass(app):
    if False:
        i = 10
        return i + 15

    class ParentException(Exception):
        pass

    class ChildExceptionUnregistered(ParentException):
        pass

    class ChildExceptionRegistered(ParentException):
        pass

    @app.errorhandler(ParentException)
    def parent_exception_handler(e):
        if False:
            i = 10
            return i + 15
        assert isinstance(e, ParentException)
        return 'parent'

    @app.errorhandler(ChildExceptionRegistered)
    def child_exception_handler(e):
        if False:
            i = 10
            return i + 15
        assert isinstance(e, ChildExceptionRegistered)
        return 'child-registered'

    @app.route('/parent')
    def parent_test():
        if False:
            return 10
        raise ParentException()

    @app.route('/child-unregistered')
    def unregistered_test():
        if False:
            return 10
        raise ChildExceptionUnregistered()

    @app.route('/child-registered')
    def registered_test():
        if False:
            i = 10
            return i + 15
        raise ChildExceptionRegistered()
    c = app.test_client()
    assert c.get('/parent').data == b'parent'
    assert c.get('/child-unregistered').data == b'parent'
    assert c.get('/child-registered').data == b'child-registered'

def test_error_handler_http_subclass(app):
    if False:
        i = 10
        return i + 15

    class ForbiddenSubclassRegistered(Forbidden):
        pass

    class ForbiddenSubclassUnregistered(Forbidden):
        pass

    @app.errorhandler(403)
    def code_exception_handler(e):
        if False:
            print('Hello World!')
        assert isinstance(e, Forbidden)
        return 'forbidden'

    @app.errorhandler(ForbiddenSubclassRegistered)
    def subclass_exception_handler(e):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(e, ForbiddenSubclassRegistered)
        return 'forbidden-registered'

    @app.route('/forbidden')
    def forbidden_test():
        if False:
            return 10
        raise Forbidden()

    @app.route('/forbidden-registered')
    def registered_test():
        if False:
            print('Hello World!')
        raise ForbiddenSubclassRegistered()

    @app.route('/forbidden-unregistered')
    def unregistered_test():
        if False:
            while True:
                i = 10
        raise ForbiddenSubclassUnregistered()
    c = app.test_client()
    assert c.get('/forbidden').data == b'forbidden'
    assert c.get('/forbidden-unregistered').data == b'forbidden'
    assert c.get('/forbidden-registered').data == b'forbidden-registered'

def test_error_handler_blueprint(app):
    if False:
        print('Hello World!')
    bp = flask.Blueprint('bp', __name__)

    @bp.errorhandler(500)
    def bp_exception_handler(e):
        if False:
            i = 10
            return i + 15
        return 'bp-error'

    @bp.route('/error')
    def bp_test():
        if False:
            while True:
                i = 10
        raise InternalServerError()

    @app.errorhandler(500)
    def app_exception_handler(e):
        if False:
            print('Hello World!')
        return 'app-error'

    @app.route('/error')
    def app_test():
        if False:
            for i in range(10):
                print('nop')
        raise InternalServerError()
    app.register_blueprint(bp, url_prefix='/bp')
    c = app.test_client()
    assert c.get('/error').data == b'app-error'
    assert c.get('/bp/error').data == b'bp-error'

def test_default_error_handler():
    if False:
        i = 10
        return i + 15
    bp = flask.Blueprint('bp', __name__)

    @bp.errorhandler(HTTPException)
    def bp_exception_handler(e):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(e, HTTPException)
        assert isinstance(e, NotFound)
        return 'bp-default'

    @bp.errorhandler(Forbidden)
    def bp_forbidden_handler(e):
        if False:
            while True:
                i = 10
        assert isinstance(e, Forbidden)
        return 'bp-forbidden'

    @bp.route('/undefined')
    def bp_registered_test():
        if False:
            i = 10
            return i + 15
        raise NotFound()

    @bp.route('/forbidden')
    def bp_forbidden_test():
        if False:
            i = 10
            return i + 15
        raise Forbidden()
    app = flask.Flask(__name__)

    @app.errorhandler(HTTPException)
    def catchall_exception_handler(e):
        if False:
            i = 10
            return i + 15
        assert isinstance(e, HTTPException)
        assert isinstance(e, NotFound)
        return 'default'

    @app.errorhandler(Forbidden)
    def catchall_forbidden_handler(e):
        if False:
            return 10
        assert isinstance(e, Forbidden)
        return 'forbidden'

    @app.route('/forbidden')
    def forbidden():
        if False:
            print('Hello World!')
        raise Forbidden()

    @app.route('/slash/')
    def slash():
        if False:
            return 10
        return 'slash'
    app.register_blueprint(bp, url_prefix='/bp')
    c = app.test_client()
    assert c.get('/bp/undefined').data == b'bp-default'
    assert c.get('/bp/forbidden').data == b'bp-forbidden'
    assert c.get('/undefined').data == b'default'
    assert c.get('/forbidden').data == b'forbidden'
    assert c.get('/slash', follow_redirects=True).data == b'slash'

class TestGenericHandlers:
    """Test how very generic handlers are dispatched to."""

    class Custom(Exception):
        pass

    @pytest.fixture()
    def app(self, app):
        if False:
            while True:
                i = 10

        @app.route('/custom')
        def do_custom():
            if False:
                for i in range(10):
                    print('nop')
            raise self.Custom()

        @app.route('/error')
        def do_error():
            if False:
                while True:
                    i = 10
            raise KeyError()

        @app.route('/abort')
        def do_abort():
            if False:
                for i in range(10):
                    print('nop')
            flask.abort(500)

        @app.route('/raise')
        def do_raise():
            if False:
                print('Hello World!')
            raise InternalServerError()
        app.config['PROPAGATE_EXCEPTIONS'] = False
        return app

    def report_error(self, e):
        if False:
            i = 10
            return i + 15
        original = getattr(e, 'original_exception', None)
        if original is not None:
            return f'wrapped {type(original).__name__}'
        return f'direct {type(e).__name__}'

    @pytest.mark.parametrize('to_handle', (InternalServerError, 500))
    def test_handle_class_or_code(self, app, client, to_handle):
        if False:
            i = 10
            return i + 15
        '``InternalServerError`` and ``500`` are aliases, they should\n        have the same behavior. Both should only receive\n        ``InternalServerError``, which might wrap another error.\n        '

        @app.errorhandler(to_handle)
        def handle_500(e):
            if False:
                print('Hello World!')
            assert isinstance(e, InternalServerError)
            return self.report_error(e)
        assert client.get('/custom').data == b'wrapped Custom'
        assert client.get('/error').data == b'wrapped KeyError'
        assert client.get('/abort').data == b'direct InternalServerError'
        assert client.get('/raise').data == b'direct InternalServerError'

    def test_handle_generic_http(self, app, client):
        if False:
            while True:
                i = 10
        '``HTTPException`` should only receive ``HTTPException``\n        subclasses. It will receive ``404`` routing exceptions.\n        '

        @app.errorhandler(HTTPException)
        def handle_http(e):
            if False:
                i = 10
                return i + 15
            assert isinstance(e, HTTPException)
            return str(e.code)
        assert client.get('/error').data == b'500'
        assert client.get('/abort').data == b'500'
        assert client.get('/not-found').data == b'404'

    def test_handle_generic(self, app, client):
        if False:
            return 10
        'Generic ``Exception`` will handle all exceptions directly,\n        including ``HTTPExceptions``.\n        '

        @app.errorhandler(Exception)
        def handle_exception(e):
            if False:
                print('Hello World!')
            return self.report_error(e)
        assert client.get('/custom').data == b'direct Custom'
        assert client.get('/error').data == b'direct KeyError'
        assert client.get('/abort').data == b'direct InternalServerError'
        assert client.get('/not-found').data == b'direct NotFound'
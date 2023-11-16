import flask
from flask.globals import request_ctx
from flask.sessions import SessionInterface

def test_open_session_with_endpoint():
    if False:
        for i in range(10):
            print('nop')
    'If request.endpoint (or other URL matching behavior) is needed\n    while loading the session, RequestContext.match_request() can be\n    called manually.\n    '

    class MySessionInterface(SessionInterface):

        def save_session(self, app, session, response):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def open_session(self, app, request):
            if False:
                return 10
            request_ctx.match_request()
            assert request.endpoint is not None
    app = flask.Flask(__name__)
    app.session_interface = MySessionInterface()

    @app.get('/')
    def index():
        if False:
            i = 10
            return i + 15
        return 'Hello, World!'
    response = app.test_client().get('/')
    assert response.status_code == 200
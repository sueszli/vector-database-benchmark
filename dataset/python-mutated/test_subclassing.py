from io import StringIO
import flask

def test_suppressed_exception_logging():
    if False:
        return 10

    class SuppressedFlask(flask.Flask):

        def log_exception(self, exc_info):
            if False:
                i = 10
                return i + 15
            pass
    out = StringIO()
    app = SuppressedFlask(__name__)

    @app.route('/')
    def index():
        if False:
            return 10
        raise Exception('test')
    rv = app.test_client().get('/', errors_stream=out)
    assert rv.status_code == 500
    assert b'Internal Server Error' in rv.data
    assert not out.getvalue()
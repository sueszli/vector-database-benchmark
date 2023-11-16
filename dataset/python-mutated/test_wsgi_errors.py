import io
import pytest
import falcon
from falcon import testing
unicode_message = 'Unicode: \x80'

@pytest.fixture
def client():
    if False:
        print('Hello World!')
    app = falcon.App()
    tehlogger = LoggerResource()
    app.add_route('/logger', tehlogger)
    return testing.TestClient(app)

class LoggerResource:

    def on_get(self, req, resp):
        if False:
            for i in range(10):
                print('nop')
        req.log_error(unicode_message)

    def on_head(self, req, resp):
        if False:
            print('Hello World!')
        req.log_error(unicode_message.encode('utf-8'))

class TestWSGIError:

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.wsgierrors_buffer = io.BytesIO()
        self.wsgierrors = io.TextIOWrapper(self.wsgierrors_buffer, line_buffering=True, encoding='utf-8')

    def test_responder_logged_bytestring(self, client):
        if False:
            while True:
                i = 10
        client.simulate_request(path='/logger', wsgierrors=self.wsgierrors, query_string='amount=10')
        log = self.wsgierrors_buffer.getvalue()
        assert unicode_message.encode('utf-8') in log
        assert b'?amount=10' in log
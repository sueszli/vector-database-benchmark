import json
import os
import tempfile
from unittest.mock import MagicMock
import tornado.httpserver
import tornado.testing
import tornado.web
import tornado.websocket
from streamlit.runtime.forward_msg_cache import ForwardMsgCache, populate_hash_if_needed
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.routes import _DEFAULT_ALLOWED_MESSAGE_ORIGINS
from streamlit.web.server.server import HEALTH_ENDPOINT, HOST_CONFIG_ENDPOINT, MESSAGE_ENDPOINT, HealthHandler, HostConfigHandler, MessageCacheHandler, StaticFileHandler
from tests.streamlit.message_mocks import create_dataframe_msg
from tests.testutil import patch_config_options

class HealthHandlerTest(tornado.testing.AsyncHTTPTestCase):
    """Tests the /_stcore/health endpoint"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(HealthHandlerTest, self).setUp()
        self._is_healthy = True

    async def is_healthy(self):
        return (self._is_healthy, 'ok')

    def get_app(self):
        if False:
            while True:
                i = 10
        return tornado.web.Application([(f'/{HEALTH_ENDPOINT}', HealthHandler, dict(callback=self.is_healthy))])

    def test_health(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/_stcore/health')
        self.assertEqual(200, response.code)
        self.assertEqual(b'ok', response.body)
        self._is_healthy = False
        response = self.fetch('/_stcore/health')
        self.assertEqual(503, response.code)

    @patch_config_options({'server.enableXsrfProtection': False})
    def test_health_without_csrf(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/_stcore/health')
        self.assertEqual(200, response.code)
        self.assertEqual(b'ok', response.body)
        self.assertNotIn('Set-Cookie', response.headers)

    @patch_config_options({'server.enableXsrfProtection': True})
    def test_health_with_csrf(self):
        if False:
            return 10
        response = self.fetch('/_stcore/health')
        self.assertEqual(200, response.code)
        self.assertEqual(b'ok', response.body)
        self.assertIn('Set-Cookie', response.headers)

    def test_health_deprecated(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/healthz')
        self.assertEqual(response.headers['link'], f'<http://127.0.0.1:{self.get_http_port()}/_stcore/health>; rel="alternate"')
        self.assertEqual(response.headers['deprecation'], 'True')

    def test_new_health_endpoint_should_not_display_deprecation_warning(self):
        if False:
            print('Hello World!')
        response = self.fetch('/_stcore/health')
        self.assertNotIn('link', response.headers)
        self.assertNotIn('deprecation', response.headers)

class MessageCacheHandlerTest(tornado.testing.AsyncHTTPTestCase):

    def get_app(self):
        if False:
            return 10
        self._cache = ForwardMsgCache()
        return tornado.web.Application([(f'/{MESSAGE_ENDPOINT}', MessageCacheHandler, dict(cache=self._cache))])

    def test_message_cache(self):
        if False:
            while True:
                i = 10
        msg = create_dataframe_msg([1, 2, 3])
        msg_hash = populate_hash_if_needed(msg)
        self._cache.add_message(msg, MagicMock(), 0)
        response = self.fetch('/_stcore/message?hash=%s' % msg_hash)
        self.assertEqual(200, response.code)
        self.assertEqual(serialize_forward_msg(msg), response.body)
        self.assertEqual(404, self.fetch('/_stcore/message').code)
        self.assertEqual(404, self.fetch('/_stcore/message?id=non_existent').code)

class StaticFileHandlerTest(tornado.testing.AsyncHTTPTestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmpfile = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, delete=False)
        self._filename = os.path.basename(self._tmpfile.name)
        super().setUp()

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self._tmpdir.cleanup()

    def get_pages(self):
        if False:
            for i in range(10):
                print('nop')
        return {'page1': 'page_info1', 'page2': 'page_info2'}

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')
        return tornado.web.Application([('/(.*)', StaticFileHandler, {'path': self._tmpdir.name, 'default_filename': self._filename, 'get_pages': self.get_pages})])

    def test_parse_url_path_200(self):
        if False:
            print('Hello World!')
        responses = [self.fetch('/'), self.fetch(f'/{self._filename}'), self.fetch('/page1/'), self.fetch(f'/page1/{self._filename}'), self.fetch('/page2/'), self.fetch(f'/page2/{self._filename}')]
        for r in responses:
            assert r.code == 200

    def test_parse_url_path_404(self):
        if False:
            for i in range(10):
                print('nop')
        responses = [self.fetch('/nonexistent'), self.fetch('/page2/nonexistent'), self.fetch(f'/page3/{self._filename}')]
        for r in responses:
            assert r.code == 404

class HostConfigHandlerTest(tornado.testing.AsyncHTTPTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(HostConfigHandlerTest, self).setUp()

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')
        return tornado.web.Application([(f'/{HOST_CONFIG_ENDPOINT}', HostConfigHandler)])

    @patch_config_options({'global.developmentMode': False})
    def test_allowed_message_origins(self):
        if False:
            print('Hello World!')
        response = self.fetch('/_stcore/host-config')
        response_body = json.loads(response.body)
        self.assertEqual(200, response.code)
        self.assertEqual({'allowedOrigins': _DEFAULT_ALLOWED_MESSAGE_ORIGINS, 'useExternalAuthToken': False, 'enableCustomParentMessages': False}, response_body)
        self.assertNotIn('http://localhost', response_body['allowedOrigins'])

    @patch_config_options({'global.developmentMode': True})
    def test_allowed_message_origins_dev_mode(self):
        if False:
            print('Hello World!')
        response = self.fetch('/_stcore/host-config')
        self.assertEqual(200, response.code)
        origins_list = json.loads(response.body)['allowedOrigins']
        self.assertIn('http://localhost', origins_list)
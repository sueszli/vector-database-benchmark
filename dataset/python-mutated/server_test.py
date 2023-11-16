import functools
import os
import multiprocessing
import shutil
import signal
import time
import tempfile
from helpers import unittest, skipOnTravisAndGithubActions
import luigi.rpc
import luigi.server
import luigi.cmdline
from luigi.configuration import get_config
from luigi.scheduler import Scheduler
from urllib.parse import urlencode, ParseResult, quote as urlquote
import tornado.ioloop
from tornado.testing import AsyncHTTPTestCase
import pytest
try:
    from unittest import mock
except ImportError:
    import mock

def _is_running_from_main_thread():
    if False:
        return 10
    '\n    Return true if we\'re the same thread as the one that created the Tornado\n    IOLoop. In practice, the problem is that we get annoying intermittent\n    failures because sometimes the KeepAliveThread jumps in and "disturbs" the\n    intended flow of the test case. Worse, it fails in the terrible way that\n    the KeepAliveThread is kept alive, bugging the execution of subsequent test\n    casses.\n\n    Oh, I so wish Tornado would explicitly say that you\'re acessing it from\n    different threads and things will just not work.\n    '
    return tornado.ioloop.IOLoop.current(instance=False)

class ServerTestBase(AsyncHTTPTestCase):

    def get_app(self):
        if False:
            return 10
        return luigi.server.app(Scheduler())

    def setUp(self):
        if False:
            while True:
                i = 10
        super(ServerTestBase, self).setUp()
        self._old_fetch = luigi.rpc.RemoteScheduler._fetch

        def _fetch(obj, url, body, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if _is_running_from_main_thread():
                body = urlencode(body).encode('utf-8')
                response = self.fetch(url, body=body, method='POST')
                if response.code >= 400:
                    raise luigi.rpc.RPCError('Errror when connecting to remote scheduler')
                return response.body.decode('utf-8')
        luigi.rpc.RemoteScheduler._fetch = _fetch

    def tearDown(self):
        if False:
            print('Hello World!')
        super(ServerTestBase, self).tearDown()
        luigi.rpc.RemoteScheduler._fetch = self._old_fetch

class ServerTest(ServerTestBase):

    def setUp(self):
        if False:
            return 10
        super(ServerTest, self).setUp()
        get_config().remove_section('cors')
        self._default_cors = luigi.server.cors()
        get_config().set('cors', 'enabled', 'true')
        get_config().set('cors', 'allow_any_origin', 'true')
        get_config().set('cors', 'allow_null_origin', 'true')

    def tearDown(self):
        if False:
            return 10
        super(ServerTest, self).tearDown()
        get_config().remove_section('cors')

    def test_visualiser(self):
        if False:
            i = 10
            return i + 15
        page = self.fetch('/').body
        self.assertTrue(page.find(b'<title>') != -1)

    def _test_404(self, path):
        if False:
            i = 10
            return i + 15
        response = self.fetch(path)
        self.assertEqual(response.code, 404)

    def test_404(self):
        if False:
            return 10
        self._test_404('/foo')

    def test_api_404(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_404('/api/foo')

    def test_root_redirect(self):
        if False:
            return 10
        response = self.fetch('/', follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers['Location'], 'static/visualiser/index.html')

    def test_api_preflight_cors_headers(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertEqual(self._default_cors.allowed_headers, headers['Access-Control-Allow-Headers'])
        self.assertEqual(self._default_cors.allowed_methods, headers['Access-Control-Allow-Methods'])
        self.assertEqual('*', headers['Access-Control-Allow-Origin'])
        self.assertEqual(str(self._default_cors.max_age), headers['Access-Control-Max-Age'])
        self.assertIsNone(headers.get('Access-Control-Allow-Credentials'))
        self.assertIsNone(headers.get('Access-Control-Expose-Headers'))

    def test_api_preflight_cors_headers_all_response_headers(self):
        if False:
            while True:
                i = 10
        get_config().set('cors', 'allow_credentials', 'true')
        get_config().set('cors', 'exposed_headers', 'foo, bar')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertEqual(self._default_cors.allowed_headers, headers['Access-Control-Allow-Headers'])
        self.assertEqual(self._default_cors.allowed_methods, headers['Access-Control-Allow-Methods'])
        self.assertEqual('*', headers['Access-Control-Allow-Origin'])
        self.assertEqual(str(self._default_cors.max_age), headers['Access-Control-Max-Age'])
        self.assertEqual('true', headers['Access-Control-Allow-Credentials'])
        self.assertEqual('foo, bar', headers['Access-Control-Expose-Headers'])

    def test_api_preflight_cors_headers_null_origin(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'null'})
        headers = dict(response.headers)
        self.assertEqual(self._default_cors.allowed_headers, headers['Access-Control-Allow-Headers'])
        self.assertEqual(self._default_cors.allowed_methods, headers['Access-Control-Allow-Methods'])
        self.assertEqual('null', headers['Access-Control-Allow-Origin'])
        self.assertEqual(str(self._default_cors.max_age), headers['Access-Control-Max-Age'])
        self.assertIsNone(headers.get('Access-Control-Allow-Credentials'))
        self.assertIsNone(headers.get('Access-Control-Expose-Headers'))

    def test_api_preflight_cors_headers_disallow_null(self):
        if False:
            i = 10
            return i + 15
        get_config().set('cors', 'allow_null_origin', 'false')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'null'})
        headers = dict(response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', headers)
        self.assertNotIn('Access-Control-Allow-Methods', headers)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertNotIn('Access-Control-Max-Age', headers)
        self.assertNotIn('Access-Control-Allow-Credentials', headers)
        self.assertNotIn('Access-Control-Expose-Headers', headers)

    def test_api_preflight_cors_headers_disallow_any(self):
        if False:
            for i in range(10):
                print('nop')
        get_config().set('cors', 'allow_any_origin', 'false')
        get_config().set('cors', 'allowed_origins', '["foo", "bar"]')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertEqual(self._default_cors.allowed_headers, headers['Access-Control-Allow-Headers'])
        self.assertEqual(self._default_cors.allowed_methods, headers['Access-Control-Allow-Methods'])
        self.assertEqual('foo', headers['Access-Control-Allow-Origin'])
        self.assertEqual(str(self._default_cors.max_age), headers['Access-Control-Max-Age'])
        self.assertIsNone(headers.get('Access-Control-Allow-Credentials'))
        self.assertIsNone(headers.get('Access-Control-Expose-Headers'))

    def test_api_preflight_cors_headers_disallow_any_no_matched_allowed_origins(self):
        if False:
            for i in range(10):
                print('nop')
        get_config().set('cors', 'allow_any_origin', 'false')
        get_config().set('cors', 'allowed_origins', '["foo", "bar"]')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foobar'})
        headers = dict(response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', headers)
        self.assertNotIn('Access-Control-Allow-Methods', headers)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertNotIn('Access-Control-Max-Age', headers)
        self.assertNotIn('Access-Control-Allow-Credentials', headers)
        self.assertNotIn('Access-Control-Expose-Headers', headers)

    def test_api_preflight_cors_headers_disallow_any_no_allowed_origins(self):
        if False:
            for i in range(10):
                print('nop')
        get_config().set('cors', 'allow_any_origin', 'false')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', headers)
        self.assertNotIn('Access-Control-Allow-Methods', headers)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertNotIn('Access-Control-Max-Age', headers)
        self.assertNotIn('Access-Control-Allow-Credentials', headers)
        self.assertNotIn('Access-Control-Expose-Headers', headers)

    def test_api_preflight_cors_headers_disabled(self):
        if False:
            i = 10
            return i + 15
        get_config().set('cors', 'enabled', 'false')
        response = self.fetch('/api/graph', method='OPTIONS', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', headers)
        self.assertNotIn('Access-Control-Allow-Methods', headers)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertNotIn('Access-Control-Max-Age', headers)
        self.assertNotIn('Access-Control-Allow-Credentials', headers)
        self.assertNotIn('Access-Control-Expose-Headers', headers)

    def test_api_preflight_cors_headers_no_origin_header(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/api/graph', method='OPTIONS')
        headers = dict(response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', headers)
        self.assertNotIn('Access-Control-Allow-Methods', headers)
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        self.assertNotIn('Access-Control-Max-Age', headers)
        self.assertNotIn('Access-Control-Allow-Credentials', headers)
        self.assertNotIn('Access-Control-Expose-Headers', headers)

    def test_api_cors_headers(self):
        if False:
            return 10
        response = self.fetch('/api/graph', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertEqual('*', headers['Access-Control-Allow-Origin'])

    def test_api_cors_headers_null_origin(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/api/graph', headers={'Origin': 'null'})
        headers = dict(response.headers)
        self.assertEqual('null', headers['Access-Control-Allow-Origin'])

    def test_api_cors_headers_disallow_null(self):
        if False:
            return 10
        get_config().set('cors', 'allow_null_origin', 'false')
        response = self.fetch('/api/graph', headers={'Origin': 'null'})
        headers = dict(response.headers)
        self.assertIsNone(headers.get('Access-Control-Allow-Origin'))

    def test_api_cors_headers_disallow_any(self):
        if False:
            while True:
                i = 10
        get_config().set('cors', 'allow_any_origin', 'false')
        get_config().set('cors', 'allowed_origins', '["foo", "bar"]')
        response = self.fetch('/api/graph', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertEqual('foo', headers['Access-Control-Allow-Origin'])

    def test_api_cors_headers_disallow_any_no_matched_allowed_origins(self):
        if False:
            print('Hello World!')
        get_config().set('cors', 'allow_any_origin', 'false')
        get_config().set('cors', 'allowed_origins', '["foo", "bar"]')
        response = self.fetch('/api/graph', headers={'Origin': 'foobar'})
        headers = dict(response.headers)
        self.assertIsNone(headers.get('Access-Control-Allow-Origin'))

    def test_api_cors_headers_disallow_any_no_allowed_origins(self):
        if False:
            for i in range(10):
                print('nop')
        get_config().set('cors', 'allow_any_origin', 'false')
        response = self.fetch('/api/graph', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertIsNone(headers.get('Access-Control-Allow-Origin'))

    def test_api_cors_headers_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        get_config().set('cors', 'enabled', 'false')
        response = self.fetch('/api/graph', headers={'Origin': 'foo'})
        headers = dict(response.headers)
        self.assertIsNone(headers.get('Access-Control-Allow-Origin'))

    def test_api_cors_headers_no_origin_header(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/api/graph')
        headers = dict(response.headers)
        self.assertIsNone(headers.get('Access-Control-Allow-Origin'))

    def test_api_allow_head_on_root(self):
        if False:
            print('Hello World!')
        response = self.fetch('/', method='HEAD')
        self.assertEqual(response.code, 204)

class _ServerTest(unittest.TestCase):
    """
    Test to start and stop the server in a more "standard" way
    """
    server_client_class = 'To be defined by subclasses'

    def start_server(self):
        if False:
            i = 10
            return i + 15
        self._process = multiprocessing.Process(target=self.server_client.run_server)
        self._process.start()
        time.sleep(0.1)
        self.sch = self.server_client.scheduler()
        self.sch._wait = lambda : None

    def stop_server(self):
        if False:
            i = 10
            return i + 15
        self._process.terminate()
        self._process.join(timeout=1)
        if self._process.is_alive():
            os.kill(self._process.pid, signal.SIGKILL)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.server_client = self.server_client_class()
        state_path = tempfile.mktemp(suffix=self.id())
        self.addCleanup(functools.partial(os.unlink, state_path))
        luigi.configuration.get_config().set('scheduler', 'state_path', state_path)
        self.start_server()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.stop_server()

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/78315794')
    def test_ping(self):
        if False:
            i = 10
            return i + 15
        self.sch.ping(worker='xyz')

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/78023665')
    def test_raw_ping(self):
        if False:
            for i in range(10):
                print('nop')
        self.sch._request('/api/ping', {'worker': 'xyz'})

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/78023665')
    def test_raw_ping_extended(self):
        if False:
            i = 10
            return i + 15
        self.sch._request('/api/ping', {'worker': 'xyz', 'foo': 'bar'})

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/166833694')
    def test_404(self):
        if False:
            return 10
        with self.assertRaises(luigi.rpc.RPCError):
            self.sch._request('/api/fdsfds', {'dummy': 1})

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/72953884')
    def test_save_state(self):
        if False:
            return 10
        self.sch.add_task(worker='X', task_id='B', deps=('A',))
        self.sch.add_task(worker='X', task_id='A')
        self.assertEqual(self.sch.get_work(worker='X')['task_id'], 'A')
        self.stop_server()
        self.start_server()
        work = self.sch.get_work(worker='X')['running_tasks'][0]
        self.assertEqual(work['task_id'], 'A')

@pytest.mark.unixsocket
class UNIXServerTest(_ServerTest):

    class ServerClient:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.tempdir = tempfile.mkdtemp()
            self.unix_socket = os.path.join(self.tempdir, 'luigid.sock')

        def run_server(self):
            if False:
                return 10
            luigi.server.run(unix_socket=self.unix_socket)

        def scheduler(self):
            if False:
                print('Hello World!')
            url = ParseResult(scheme='http+unix', netloc=urlquote(self.unix_socket, safe=''), path='', params='', query='', fragment='').geturl()
            return luigi.rpc.RemoteScheduler(url)
    server_client_class = ServerClient

    def tearDown(self):
        if False:
            while True:
                i = 10
        super(UNIXServerTest, self).tearDown()
        shutil.rmtree(self.server_client.tempdir)

class INETServerClient:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.port = 8083

    def scheduler(self):
        if False:
            for i in range(10):
                print('nop')
        return luigi.rpc.RemoteScheduler('http://localhost:' + str(self.port))

class _INETServerTest(_ServerTest):
    __test__ = False

    def test_with_cmdline(self):
        if False:
            print('Hello World!')
        '\n        Test to run against the server as a normal luigi invocation does\n        '
        params = ['Task', '--scheduler-port', str(self.server_client.port), '--no-lock']
        self.assertTrue(luigi.interface.run(params))

class INETProcessServerTest(_INETServerTest):
    __test__ = True

    class ServerClient(INETServerClient):

        def run_server(self):
            if False:
                print('Hello World!')
            luigi.server.run(api_port=self.port, address='127.0.0.1')
    server_client_class = ServerClient

class INETURLLibServerTest(INETProcessServerTest):

    @mock.patch.object(luigi.rpc, 'HAS_REQUESTS', False)
    def start_server(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(INETURLLibServerTest, self).start_server(*args, **kwargs)

    @skipOnTravisAndGithubActions('https://travis-ci.org/spotify/luigi/jobs/81022689')
    def patching_test(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that HAS_REQUESTS patching is meaningful\n        '
        fetcher1 = luigi.rpc.RemoteScheduler()._fetcher
        with mock.patch.object(luigi.rpc, 'HAS_REQUESTS', False):
            fetcher2 = luigi.rpc.RemoteScheduler()._fetcher
        self.assertNotEqual(fetcher1.__class__, fetcher2.__class__)

class INETLuigidServerTest(_INETServerTest):
    __test__ = True

    class ServerClient(INETServerClient):

        def run_server(self):
            if False:
                return 10
            luigi.cmdline.luigid(['--port', str(self.port)])
    server_client_class = ServerClient

class INETLuigidDaemonServerTest(_INETServerTest):
    __test__ = True

    class ServerClient(INETServerClient):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super(INETLuigidDaemonServerTest.ServerClient, self).__init__()
            self.tempdir = tempfile.mkdtemp()

        @mock.patch('daemon.DaemonContext')
        def run_server(self, daemon_context):
            if False:
                print('Hello World!')
            luigi.cmdline.luigid(['--port', str(self.port), '--background', '--logdir', self.tempdir, '--pidfile', os.path.join(self.tempdir, 'luigid.pid')])

    def tearDown(self):
        if False:
            return 10
        super(INETLuigidDaemonServerTest, self).tearDown()
        shutil.rmtree(self.server_client.tempdir)
    server_client_class = ServerClient

class MetricsHandlerTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.mock_scheduler = mock.MagicMock()
        self.handler = luigi.server.MetricsHandler(tornado.web.Application(), mock.MagicMock(), scheduler=self.mock_scheduler)

    def test_initialize(self):
        if False:
            return 10
        self.assertIs(self.handler._scheduler, self.mock_scheduler)

    def test_get(self):
        if False:
            print('Hello World!')
        mock_metrics = mock.MagicMock()
        self.mock_scheduler._state._metrics_collector.generate_latest.return_value = mock_metrics
        with mock.patch.object(self.handler, 'write') as patched_write:
            self.handler.get()
            patched_write.assert_called_once_with(mock_metrics)
            self.mock_scheduler._state._metrics_collector.configure_http_handler.assert_called_once_with(self.handler)

    def test_get_no_metrics(self):
        if False:
            i = 10
            return i + 15
        self.mock_scheduler._state._metrics_collector.generate_latest.return_value = None
        with mock.patch.object(self.handler, 'write') as patched_write:
            self.handler.get()
            patched_write.assert_not_called()
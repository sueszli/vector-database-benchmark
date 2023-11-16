"""End-to-end tests for Python profiler wrapper."""
# pylint: disable=missing-docstring, blacklisted-name
import functools
import gzip
import json
import inspect
import threading
import time
import unittest
import urllib.request

from vprof import profiler
from vprof import stats_server
from vprof import runner
from vprof.tests import test_pkg # pylint: disable=unused-import

_HOST, _PORT = 'localhost', 12345
_MODULE_FILENAME = 'vprof/tests/test_pkg/dummy_module.py'
_PACKAGE_PATH = 'vprof/tests/test_pkg/'
_POLL_INTERVAL = 0.01


class ProfilerModuleEndToEndTest(unittest.TestCase):

    def setUp(self):
        program_stats = profiler.Profiler(
            _MODULE_FILENAME).run()
        stats_handler = functools.partial(
            stats_server.StatsHandler, program_stats)
        self.server = stats_server.StatsServer(
            (_HOST, _PORT), stats_handler)
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={'poll_interval': _POLL_INTERVAL}).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def testRequest(self):
        response = urllib.request.urlopen(
            'http://%s:%s/profile' % (_HOST, _PORT))
        response_data = gzip.decompress(response.read())
        stats = json.loads(response_data.decode('utf-8'))
        self.assertEqual(stats['objectName'], '%s (module)' % _MODULE_FILENAME)
        self.assertTrue(len(stats['callStats']) > 0)
        self.assertTrue(stats['totalTime'] > 0)
        self.assertTrue(stats['primitiveCalls'] > 0)
        self.assertTrue(stats['totalCalls'] > 0)


class ProfilerPackageEndToEndTest(unittest.TestCase):

    def setUp(self):
        program_stats = profiler.Profiler(
            _PACKAGE_PATH).run()
        stats_handler = functools.partial(
            stats_server.StatsHandler, program_stats)
        self.server = stats_server.StatsServer(
            (_HOST, _PORT), stats_handler)
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={'poll_interval': _POLL_INTERVAL}).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def testRequest(self):
        response = urllib.request.urlopen(
            'http://%s:%s/profile' % (_HOST, _PORT))
        response_data = gzip.decompress(response.read())
        stats = json.loads(response_data.decode('utf-8'))
        self.assertEqual(stats['objectName'], '%s (package)' % _PACKAGE_PATH)
        self.assertTrue(len(stats['callStats']) > 0)
        self.assertTrue(stats['totalTime'] > 0)
        self.assertTrue(stats['primitiveCalls'] > 0)
        self.assertTrue(stats['totalCalls'] > 0)


class ProfilerFunctionEndToEndTest(unittest.TestCase):

    def setUp(self):

        def _func(foo, bar):
            baz = foo + bar
            time.sleep(0.1)
            return baz
        self._func = _func

        stats_handler = functools.partial(
            stats_server.StatsHandler, {})
        self.server = stats_server.StatsServer(
            (_HOST, _PORT), stats_handler)
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={'poll_interval': _POLL_INTERVAL}).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def testRequest(self):
        runner.run(
            self._func, 'p', ('foo', 'bar'), host=_HOST, port=_PORT)
        response = urllib.request.urlopen(
            'http://%s:%s/profile' % (_HOST, _PORT))
        response_data = gzip.decompress(response.read())
        stats = json.loads(response_data.decode('utf-8'))
        curr_filename = inspect.getabsfile(inspect.currentframe())
        self.assertEqual(stats['p']['objectName'],
                         '_func @ %s (function)' % curr_filename)
        self.assertTrue(len(stats['p']['callStats']) > 0)
        self.assertTrue(stats['p']['totalTime'] > 0)
        self.assertTrue(stats['p']['primitiveCalls'] > 0)
        self.assertTrue(stats['p']['totalCalls'] > 0)


# pylint: enable=missing-docstring, blacklisted-name

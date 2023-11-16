"""Glances unitary tests suite for the RESTful API."""
import os
import shlex
import subprocess
import time
import numbers
import unittest
from glances import __version__
from glances.globals import text_type
import requests
SERVER_PORT = 61234
API_VERSION = 3
URL = 'http://localhost:{}/api/{}'.format(SERVER_PORT, API_VERSION)
pid = None
print('RESTful API unitary tests for Glances %s' % __version__)

class TestGlances(unittest.TestCase):
    """Test Glances class."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'The function is called *every time* before test_*.'
        print('\n' + '=' * 78)

    def http_get(self, url, deflate=False):
        if False:
            for i in range(10):
                print('nop')
        'Make the request'
        if deflate:
            ret = requests.get(url, stream=True, headers={'Accept-encoding': 'deflate'})
        else:
            ret = requests.get(url, headers={'Accept-encoding': 'identity'})
        return ret

    def test_000_start_server(self):
        if False:
            print('Hello World!')
        'Start the Glances Web Server.'
        global pid
        print('INFO: [TEST_000] Start the Glances Web Server')
        if os.path.isfile('./venv/bin/python'):
            cmdline = './venv/bin/python'
        else:
            cmdline = 'python'
        cmdline += ' -m glances -B localhost -w -p %s' % SERVER_PORT
        print('Run the Glances Web Server on port %s' % SERVER_PORT)
        args = shlex.split(cmdline)
        pid = subprocess.Popen(args)
        print('Please wait 5 seconds...')
        time.sleep(5)
        self.assertTrue(pid is not None)

    def test_001_all(self):
        if False:
            while True:
                i = 10
        'All.'
        method = 'all'
        print('INFO: [TEST_001] Get all stats')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)

    def test_001a_all_deflate(self):
        if False:
            while True:
                i = 10
        'All.'
        method = 'all'
        print('INFO: [TEST_001a] Get all stats (with Deflate compression)')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method), deflate=True)
        self.assertTrue(req.ok)
        self.assertTrue(req.headers['Content-Encoding'] == 'deflate')

    def test_002_pluginslist(self):
        if False:
            i = 10
            return i + 15
        'Plugins list.'
        method = 'pluginslist'
        print('INFO: [TEST_002] Plugins list')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), list)
        self.assertIn('cpu', req.json())

    def test_003_plugins(self):
        if False:
            print('Hello World!')
        'Plugins.'
        method = 'pluginslist'
        print('INFO: [TEST_003] Plugins')
        plist = self.http_get('%s/%s' % (URL, method))
        for p in plist.json():
            print('HTTP RESTful request: %s/%s' % (URL, p))
            req = self.http_get('%s/%s' % (URL, p))
            self.assertTrue(req.ok)
            if p in ('uptime', 'now'):
                self.assertIsInstance(req.json(), text_type)
            elif p in ('fs', 'percpu', 'sensors', 'alert', 'processlist', 'diskio', 'hddtemp', 'batpercent', 'network', 'folders', 'amps', 'ports', 'irq', 'wifi', 'gpu'):
                self.assertIsInstance(req.json(), list)
            elif p in ('psutilversion', 'help'):
                pass
            else:
                self.assertIsInstance(req.json(), dict)

    def test_004_items(self):
        if False:
            while True:
                i = 10
        'Items.'
        method = 'cpu'
        print('INFO: [TEST_004] Items for the CPU method')
        ilist = self.http_get('%s/%s' % (URL, method))
        for i in ilist.json():
            print('HTTP RESTful request: %s/%s/%s' % (URL, method, i))
            req = self.http_get('%s/%s/%s' % (URL, method, i))
            self.assertTrue(req.ok)
            self.assertIsInstance(req.json(), dict)
            print(req.json()[i])
            self.assertIsInstance(req.json()[i], numbers.Number)

    def test_005_values(self):
        if False:
            print('Hello World!')
        'Values.'
        method = 'processlist'
        print('INFO: [TEST_005] Item=Value for the PROCESSLIST method')
        print('%s/%s/pid/0' % (URL, method))
        req = self.http_get('%s/%s/pid/0' % (URL, method))
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), dict)

    def test_006_all_limits(self):
        if False:
            return 10
        'All limits.'
        method = 'all/limits'
        print('INFO: [TEST_006] Get all limits')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), dict)

    def test_007_all_views(self):
        if False:
            while True:
                i = 10
        'All views.'
        method = 'all/views'
        print('INFO: [TEST_007] Get all views')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), dict)

    def test_008_plugins_limits(self):
        if False:
            for i in range(10):
                print('nop')
        'Plugins limits.'
        method = 'pluginslist'
        print('INFO: [TEST_008] Plugins limits')
        plist = self.http_get('%s/%s' % (URL, method))
        for p in plist.json():
            print('HTTP RESTful request: %s/%s/limits' % (URL, p))
            req = self.http_get('%s/%s/limits' % (URL, p))
            self.assertTrue(req.ok)
            self.assertIsInstance(req.json(), dict)

    def test_009_plugins_views(self):
        if False:
            return 10
        'Plugins views.'
        method = 'pluginslist'
        print('INFO: [TEST_009] Plugins views')
        plist = self.http_get('%s/%s' % (URL, method))
        for p in plist.json():
            print('HTTP RESTful request: %s/%s/views' % (URL, p))
            req = self.http_get('%s/%s/views' % (URL, p))
            self.assertTrue(req.ok)
            self.assertIsInstance(req.json(), dict)

    def test_010_history(self):
        if False:
            while True:
                i = 10
        'History.'
        method = 'history'
        print('INFO: [TEST_010] History')
        print('HTTP RESTful request: %s/cpu/%s' % (URL, method))
        req = self.http_get('%s/cpu/%s' % (URL, method))
        self.assertIsInstance(req.json(), dict)
        self.assertIsInstance(req.json()['user'], list)
        self.assertTrue(len(req.json()['user']) > 0)
        print('HTTP RESTful request: %s/cpu/%s/3' % (URL, method))
        req = self.http_get('%s/cpu/%s/3' % (URL, method))
        self.assertIsInstance(req.json(), dict)
        self.assertIsInstance(req.json()['user'], list)
        self.assertTrue(len(req.json()['user']) > 1)
        print('HTTP RESTful request: %s/cpu/system/%s' % (URL, method))
        req = self.http_get('%s/cpu/system/%s' % (URL, method))
        self.assertIsInstance(req.json(), dict)
        self.assertIsInstance(req.json()['system'], list)
        self.assertTrue(len(req.json()['system']) > 0)
        print('HTTP RESTful request: %s/cpu/system/%s/3' % (URL, method))
        req = self.http_get('%s/cpu/system/%s/3' % (URL, method))
        self.assertIsInstance(req.json(), dict)
        self.assertIsInstance(req.json()['system'], list)
        self.assertTrue(len(req.json()['system']) > 1)

    def test_011_issue1401(self):
        if False:
            i = 10
            return i + 15
        'Check issue #1401.'
        method = 'network/interface_name'
        print('INFO: [TEST_011] Issue #1401')
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), dict)
        self.assertIsInstance(req.json()['interface_name'], list)

    def test_012_status(self):
        if False:
            i = 10
            return i + 15
        'Check status endpoint.'
        method = 'status'
        print('INFO: [TEST_012] Status')
        print('HTTP RESTful request: %s/%s' % (URL, method))
        req = self.http_get('%s/%s' % (URL, method))
        self.assertTrue(req.ok)
        self.assertEqual(req.text, 'Active')

    def test_013_top(self):
        if False:
            return 10
        'Values.'
        method = 'processlist'
        request = '%s/%s/top/2' % (URL, method)
        print('INFO: [TEST_013] Top nb item of PROCESSLIST')
        print(request)
        req = self.http_get(request)
        self.assertTrue(req.ok)
        self.assertIsInstance(req.json(), list)
        self.assertEqual(len(req.json()), 2)

    def test_999_stop_server(self):
        if False:
            for i in range(10):
                print('nop')
        'Stop the Glances Web Server.'
        print('INFO: [TEST_999] Stop the Glances Web Server')
        print('Stop the Glances Web Server')
        pid.terminate()
        time.sleep(1)
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()
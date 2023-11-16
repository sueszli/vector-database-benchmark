"""Glances unitary tests suite for the XML-RPC API."""
import os
import json
import shlex
import subprocess
import time
import unittest
from glances import __version__
from glances.globals import ServerProxy
SERVER_PORT = 61234
URL = 'http://localhost:%s' % SERVER_PORT
pid = None
client = ServerProxy(URL)
print('XML-RPC API unitary tests for Glances %s' % __version__)

class TestGlances(unittest.TestCase):
    """Test Glances class."""

    def setUp(self):
        if False:
            return 10
        'The function is called *every time* before test_*.'
        print('\n' + '=' * 78)

    def test_000_start_server(self):
        if False:
            i = 10
            return i + 15
        'Start the Glances Web Server.'
        global pid
        print('INFO: [TEST_000] Start the Glances Web Server')
        if os.path.isfile('./venv/bin/python'):
            cmdline = './venv/bin/python'
        else:
            cmdline = 'python'
        cmdline += ' -m glances -B localhost -s -p %s' % SERVER_PORT
        print('Run the Glances Server on port %s' % SERVER_PORT)
        args = shlex.split(cmdline)
        pid = subprocess.Popen(args)
        print('Please wait...')
        time.sleep(1)
        self.assertTrue(pid is not None)

    def test_001_all(self):
        if False:
            return 10
        'All.'
        method = 'getAll()'
        print('INFO: [TEST_001] Connection test')
        print('XML-RPC request: %s' % method)
        req = json.loads(client.getAll())
        self.assertIsInstance(req, dict)

    def test_002_pluginslist(self):
        if False:
            print('Hello World!')
        'Plugins list.'
        method = 'getAllPlugins()'
        print('INFO: [TEST_002] Get plugins list')
        print('XML-RPC request: %s' % method)
        req = json.loads(client.getAllPlugins())
        self.assertIsInstance(req, list)

    def test_003_system(self):
        if False:
            i = 10
            return i + 15
        'System.'
        method = 'getSystem()'
        print('INFO: [TEST_003] Method: %s' % method)
        req = json.loads(client.getSystem())
        self.assertIsInstance(req, dict)

    def test_004_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        'CPU.'
        method = 'getCpu(), getPerCpu(), getLoad() and getCore()'
        print('INFO: [TEST_004] Method: %s' % method)
        req = json.loads(client.getCpu())
        self.assertIsInstance(req, dict)
        req = json.loads(client.getPerCpu())
        self.assertIsInstance(req, list)
        req = json.loads(client.getLoad())
        self.assertIsInstance(req, dict)
        req = json.loads(client.getCore())
        self.assertIsInstance(req, dict)

    def test_005_mem(self):
        if False:
            while True:
                i = 10
        'MEM.'
        method = 'getMem() and getMemSwap()'
        print('INFO: [TEST_005] Method: %s' % method)
        req = json.loads(client.getMem())
        self.assertIsInstance(req, dict)
        req = json.loads(client.getMemSwap())
        self.assertIsInstance(req, dict)

    def test_006_net(self):
        if False:
            while True:
                i = 10
        'NETWORK.'
        method = 'getNetwork()'
        print('INFO: [TEST_006] Method: %s' % method)
        req = json.loads(client.getNetwork())
        self.assertIsInstance(req, list)

    def test_007_disk(self):
        if False:
            i = 10
            return i + 15
        'DISK.'
        method = 'getFs(), getFolders() and getDiskIO()'
        print('INFO: [TEST_007] Method: %s' % method)
        req = json.loads(client.getFs())
        self.assertIsInstance(req, list)
        req = json.loads(client.getFolders())
        self.assertIsInstance(req, list)
        req = json.loads(client.getDiskIO())
        self.assertIsInstance(req, list)

    def test_008_sensors(self):
        if False:
            print('Hello World!')
        'SENSORS.'
        method = 'getSensors()'
        print('INFO: [TEST_008] Method: %s' % method)
        req = json.loads(client.getSensors())
        self.assertIsInstance(req, list)

    def test_009_process(self):
        if False:
            for i in range(10):
                print('nop')
        'PROCESS.'
        method = 'getProcessCount() and getProcessList()'
        print('INFO: [TEST_009] Method: %s' % method)
        req = json.loads(client.getProcessCount())
        self.assertIsInstance(req, dict)
        req = json.loads(client.getProcessList())
        self.assertIsInstance(req, list)

    def test_010_all_limits(self):
        if False:
            for i in range(10):
                print('nop')
        'All limits.'
        method = 'getAllLimits()'
        print('INFO: [TEST_010] Method: %s' % method)
        req = json.loads(client.getAllLimits())
        self.assertIsInstance(req, dict)
        self.assertIsInstance(req['cpu'], dict)

    def test_011_all_views(self):
        if False:
            return 10
        'All views.'
        method = 'getAllViews()'
        print('INFO: [TEST_011] Method: %s' % method)
        req = json.loads(client.getAllViews())
        self.assertIsInstance(req, dict)
        self.assertIsInstance(req['cpu'], dict)

    def test_012_irq(self):
        if False:
            return 10
        'IRQS'
        method = 'getIrqs()'
        print('INFO: [TEST_012] Method: %s' % method)
        req = json.loads(client.getIrq())
        self.assertIsInstance(req, list)

    def test_013_plugin_views(self):
        if False:
            print('Hello World!')
        'Plugin views.'
        method = 'getViewsCpu()'
        print('INFO: [TEST_013] Method: %s' % method)
        req = json.loads(client.getViewsCpu())
        self.assertIsInstance(req, dict)

    def test_999_stop_server(self):
        if False:
            i = 10
            return i + 15
        'Stop the Glances Web Server.'
        print('INFO: [TEST_999] Stop the Glances Server')
        print('Stop the Glances Server')
        pid.terminate()
        time.sleep(1)
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()
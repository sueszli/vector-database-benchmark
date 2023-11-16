from __future__ import absolute_import
from __future__ import print_function
from twisted.trial import unittest
from buildbot_worker.scripts import start
from buildbot_worker.test.util import misc
try:
    from unittest import mock
except ImportError:
    import mock

class TestStartCommand(unittest.TestCase, misc.IsWorkerDirMixin):
    """
    Test buildbot_worker.scripts.startup.startCommand()
    """

    def test_start_command_bad_basedir(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test calling startCommand() with invalid basedir path\n        '
        self.setupUpIsWorkerDir(False)
        config = {'basedir': 'dummy'}
        self.assertEqual(start.startCommand(config), 1, 'unexpected exit code')
        self.isWorkerDir.assert_called_once_with('dummy')

    def test_start_command_good(self):
        if False:
            return 10
        '\n        test successful startCommand() call\n        '
        self.setupUpIsWorkerDir(True)
        mocked_startWorker = mock.Mock(return_value=0)
        self.patch(start, 'startWorker', mocked_startWorker)
        config = {'basedir': 'dummy', 'nodaemon': False, 'quiet': False}
        self.assertEqual(start.startCommand(config), 0, 'unexpected exit code')
        self.isWorkerDir.assert_called_once_with('dummy')
        mocked_startWorker.assert_called_once_with(config['basedir'], config['quiet'], config['nodaemon'])
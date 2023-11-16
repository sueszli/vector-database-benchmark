from __future__ import absolute_import
from __future__ import print_function
from twisted.trial import unittest
from buildbot_worker.scripts import restart
from buildbot_worker.scripts import start
from buildbot_worker.scripts import stop
from buildbot_worker.test.util import misc
try:
    from unittest import mock
except ImportError:
    import mock

class TestRestart(misc.IsWorkerDirMixin, misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.restart.restart()
    """
    config = {'basedir': 'dummy', 'nodaemon': False, 'quiet': False}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpStdoutAssertions()
        self.startWorker = mock.Mock()
        self.patch(start, 'startWorker', self.startWorker)

    def test_bad_basedir(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test calling restart() with invalid basedir path\n        '
        self.setupUpIsWorkerDir(False)
        self.assertEqual(restart.restart(self.config), 1, 'unexpected exit code')
        self.isWorkerDir.assert_called_once_with(self.config['basedir'])

    def test_no_worker_running(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test calling restart() when no worker is running\n        '
        self.setupUpIsWorkerDir(True)
        mock_stopWorker = mock.Mock(side_effect=stop.WorkerNotRunning())
        self.patch(stop, 'stopWorker', mock_stopWorker)
        restart.restart(self.config)
        self.startWorker.assert_called_once_with(self.config['basedir'], self.config['quiet'], self.config['nodaemon'])
        self.assertStdoutEqual('no old worker process found to stop\nnow restarting worker process..\n')

    def test_restart(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test calling restart() when worker is running\n        '
        self.setupUpIsWorkerDir(True)
        mock_stopWorker = mock.Mock()
        self.patch(stop, 'stopWorker', mock_stopWorker)
        restart.restart(self.config)
        self.startWorker.assert_called_once_with(self.config['basedir'], self.config['quiet'], self.config['nodaemon'])
        self.assertStdoutEqual('now restarting worker process..\n')
from __future__ import absolute_import
from __future__ import print_function
import errno
import os
import signal
import time
from twisted.trial import unittest
from buildbot_worker.scripts import stop
from buildbot_worker.test.util import compat
from buildbot_worker.test.util import misc
try:
    from unittest import mock
except ImportError:
    import mock

class TestStopWorker(misc.FileIOMixin, misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.stop.stopWorker()
    """
    PID = 9876

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpStdoutAssertions()
        self.patch(os, 'chdir', mock.Mock())

    def test_no_pid_file(self):
        if False:
            i = 10
            return i + 15
        '\n        test calling stopWorker() when no pid file is present\n        '
        self.setUpOpenError(2)
        with self.assertRaises(stop.WorkerNotRunning):
            stop.stopWorker(None, False)

    @compat.skipUnlessPlatformIs('posix')
    def test_successful_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test stopWorker() on a successful worker stop\n        '

        def emulated_kill(pid, sig):
            if False:
                for i in range(10):
                    print('nop')
            if sig == 0:
                raise OSError(errno.ESRCH, 'dummy')
        self.setUpOpen(str(self.PID))
        mocked_kill = mock.Mock(side_effect=emulated_kill)
        self.patch(os, 'kill', mocked_kill)
        self.patch(time, 'sleep', mock.Mock())
        exit_code = stop.stopWorker(None, False)
        self.assertEqual(exit_code, 0)
        mocked_kill.assert_has_calls([mock.call(self.PID, signal.SIGTERM), mock.call(self.PID, 0)])
        self.assertStdoutEqual('worker process {0} is dead\n'.format(self.PID))

    @compat.skipUnlessPlatformIs('posix')
    def test_stop_timeout(self):
        if False:
            return 10
        '\n        test stopWorker() when stop timeouts\n        '
        self.setUpOpen(str(self.PID))
        mocked_kill = mock.Mock()
        self.patch(os, 'kill', mocked_kill)
        self.patch(time, 'sleep', mock.Mock())
        exit_code = stop.stopWorker(None, False)
        self.assertEqual(exit_code, 1)
        mocked_kill.assert_has_calls([mock.call(self.PID, signal.SIGTERM), mock.call(self.PID, 0)])
        self.assertStdoutEqual('never saw process go away\n')

class TestStop(misc.IsWorkerDirMixin, misc.StdoutAssertionsMixin, unittest.TestCase):
    """
    Test buildbot_worker.scripts.stop.stop()
    """
    config = {'basedir': 'dummy', 'quiet': False}

    def test_bad_basedir(self):
        if False:
            while True:
                i = 10
        '\n        test calling stop() with invalid basedir path\n        '
        self.setupUpIsWorkerDir(False)
        self.assertEqual(stop.stop(self.config), 1, 'unexpected exit code')
        self.isWorkerDir.assert_called_once_with(self.config['basedir'])

    def test_no_worker_running(self):
        if False:
            return 10
        '\n        test calling stop() when no worker is running\n        '
        self.setUpStdoutAssertions()
        self.setupUpIsWorkerDir(True)
        mock_stopWorker = mock.Mock(side_effect=stop.WorkerNotRunning())
        self.patch(stop, 'stopWorker', mock_stopWorker)
        exit_code = stop.stop(self.config)
        self.assertEqual(exit_code, 0)
        self.assertStdoutEqual('worker not running\n')

    def test_successful_stop(self):
        if False:
            i = 10
            return i + 15
        '\n        test calling stop() when worker is running\n        '
        self.setupUpIsWorkerDir(True)
        mock_stopWorker = mock.Mock(return_value=0)
        self.patch(stop, 'stopWorker', mock_stopWorker)
        exit_code = stop.stop(self.config)
        self.assertEqual(exit_code, 0)
        mock_stopWorker.assert_called_once_with(self.config['basedir'], self.config['quiet'], 'TERM')

    def test_failed_stop(self):
        if False:
            while True:
                i = 10
        '\n        test failing stop()\n        '
        self.setupUpIsWorkerDir(True)
        mock_stopWorker = mock.Mock(return_value=17)
        self.patch(stop, 'stopWorker', mock_stopWorker)
        exit_code = stop.stop(self.config)
        self.assertEqual(exit_code, 17)
        mock_stopWorker.assert_called_once_with(self.config['basedir'], self.config['quiet'], 'TERM')
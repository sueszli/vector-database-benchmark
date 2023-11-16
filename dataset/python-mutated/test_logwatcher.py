import os
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.scripts.logwatcher import BuildmasterStartupError
from buildbot.scripts.logwatcher import BuildmasterTimeoutError
from buildbot.scripts.logwatcher import LogWatcher
from buildbot.scripts.logwatcher import ReconfigError
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import dirs
from buildbot.util import unicode2bytes

class MockedLogWatcher(LogWatcher):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.printed_output = []
        self.created_paths = []

    def create_logfile(self, path):
        if False:
            i = 10
            return i + 15
        self.created_paths.append(path)

    def print_output(self, output):
        if False:
            i = 10
            return i + 15
        self.printed_output.append(output)

class TestLogWatcher(unittest.TestCase, dirs.DirsMixin, TestReactorMixin):
    delimiter = unicode2bytes(os.linesep)

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpDirs('workdir')
        self.addCleanup(self.tearDownDirs)
        self.setup_test_reactor()
        self.spawned_process = mock.Mock()
        self.reactor.spawnProcess = mock.Mock(return_value=self.spawned_process)

    def test_start(self):
        if False:
            while True:
                i = 10
        lw = MockedLogWatcher('workdir/test.log', _reactor=self.reactor)
        lw._start = mock.Mock()
        lw.start()
        self.reactor.spawnProcess.assert_called()
        self.assertEqual(lw.created_paths, ['workdir/test.log'])
        self.assertTrue(lw.running)

    @defer.inlineCallbacks
    def test_success_before_timeout(self):
        if False:
            print('Hello World!')
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        self.reactor.advance(4.9)
        lw.lineReceived(b'BuildMaster is running')
        res = (yield d)
        self.assertEqual(res, 'buildmaster')

    @defer.inlineCallbacks
    def test_failure_after_timeout(self):
        if False:
            while True:
                i = 10
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        self.reactor.advance(5.1)
        lw.lineReceived(b'BuildMaster is running')
        with self.assertRaises(BuildmasterTimeoutError):
            yield d

    @defer.inlineCallbacks
    def test_progress_restarts_timeout(self):
        if False:
            while True:
                i = 10
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        self.reactor.advance(4.9)
        lw.lineReceived(b'added builder')
        self.reactor.advance(4.9)
        lw.lineReceived(b'BuildMaster is running')
        res = (yield d)
        self.assertEqual(res, 'buildmaster')

    @defer.inlineCallbacks
    def test_handles_very_long_lines(self):
        if False:
            for i in range(10):
                print('nop')
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        lw.dataReceived(b't' * lw.MAX_LENGTH * 2 + self.delimiter + b'BuildMaster is running' + self.delimiter)
        res = (yield d)
        self.assertEqual(lw.printed_output, ['Got an a very long line in the log (length 32768 bytes), ignoring'])
        self.assertEqual(res, 'buildmaster')

    @defer.inlineCallbacks
    def test_handles_very_long_lines_separate_packet(self):
        if False:
            return 10
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        lw.dataReceived(b't' * lw.MAX_LENGTH * 2)
        lw.dataReceived(self.delimiter + b'BuildMaster is running' + self.delimiter)
        res = (yield d)
        self.assertEqual(lw.printed_output, ['Got an a very long line in the log (length 32768 bytes), ignoring'])
        self.assertEqual(res, 'buildmaster')

    @defer.inlineCallbacks
    def test_handles_very_long_lines_separate_packet_with_newline(self):
        if False:
            print('Hello World!')
        lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
        d = lw.start()
        lw.dataReceived(b't' * lw.MAX_LENGTH * 2 + self.delimiter)
        lw.dataReceived(b'BuildMaster is running' + self.delimiter)
        res = (yield d)
        self.assertEqual(lw.printed_output, ['Got an a very long line in the log (length 32768 bytes), ignoring'])
        self.assertEqual(res, 'buildmaster')

    @defer.inlineCallbacks
    def test_matches_lines(self):
        if False:
            for i in range(10):
                print('nop')
        lines_and_expected = [(b'configuration update aborted without making any changes', ReconfigError()), (b'WARNING: configuration update partially applied; master may malfunction', ReconfigError()), (b'Server Shut Down', ReconfigError()), (b'BuildMaster startup failed', BuildmasterStartupError()), (b'message from master: attached', 'worker'), (b'configuration update complete', 'buildmaster'), (b'BuildMaster is running', 'buildmaster')]
        for (line, expected) in lines_and_expected:
            lw = MockedLogWatcher('workdir/test.log', timeout=5, _reactor=self.reactor)
            d = lw.start()
            lw.lineReceived(line)
            if isinstance(expected, Exception):
                with self.assertRaises(type(expected)):
                    yield d
            else:
                res = (yield d)
                self.assertEqual(res, expected)
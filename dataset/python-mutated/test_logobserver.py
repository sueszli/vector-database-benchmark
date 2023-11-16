from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process import log
from buildbot.process import logobserver
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin

class MyLogObserver(logobserver.LogObserver):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.obs = []

    def outReceived(self, data):
        if False:
            while True:
                i = 10
        self.obs.append(('out', data))

    def errReceived(self, data):
        if False:
            i = 10
            return i + 15
        self.obs.append(('err', data))

    def headerReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.obs.append(('hdr', data))

    def finishReceived(self):
        if False:
            for i in range(10):
                print('nop')
        self.obs.append(('fin',))

class TestLogObserver(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)

    @defer.inlineCallbacks
    def test_sequence(self):
        if False:
            i = 10
            return i + 15
        logid = (yield self.master.data.updates.addLog(1, 'mine', 's'))
        _log = log.Log.new(self.master, 'mine', 's', logid, 'utf-8')
        lo = MyLogObserver()
        lo.setLog(_log)
        yield _log.addStdout('hello\n')
        yield _log.addStderr('cruel\n')
        yield _log.addStdout('world\n')
        yield _log.addStdout('multi\nline\nchunk\n')
        yield _log.addHeader('HDR\n')
        yield _log.finish()
        self.assertEqual(lo.obs, [('out', 'hello\n'), ('err', 'cruel\n'), ('out', 'world\n'), ('out', 'multi\nline\nchunk\n'), ('hdr', 'HDR\n'), ('fin',)])

class MyLogLineObserver(logobserver.LogLineObserver):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.obs = []

    def outLineReceived(self, line):
        if False:
            print('Hello World!')
        self.obs.append(('out', line))

    def errLineReceived(self, line):
        if False:
            return 10
        self.obs.append(('err', line))

    def headerLineReceived(self, line):
        if False:
            print('Hello World!')
        self.obs.append(('hdr', line))

    def finishReceived(self):
        if False:
            for i in range(10):
                print('nop')
        self.obs.append(('fin',))

class TestLineConsumerLogObesrver(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)

    @defer.inlineCallbacks
    def do_test_sequence(self, consumer):
        if False:
            while True:
                i = 10
        logid = (yield self.master.data.updates.addLog(1, 'mine', 's'))
        _log = log.Log.new(self.master, 'mine', 's', logid, 'utf-8')
        lo = logobserver.LineConsumerLogObserver(consumer)
        lo.setLog(_log)
        yield _log.addStdout('hello\n')
        yield _log.addStderr('cruel\n')
        yield _log.addStdout('multi\nline\nchunk\n')
        yield _log.addHeader('H1\nH2\n')
        yield _log.finish()

    @defer.inlineCallbacks
    def test_sequence_finish(self):
        if False:
            while True:
                i = 10
        results = []

        def consumer():
            if False:
                print('Hello World!')
            while True:
                try:
                    (stream, line) = (yield)
                    results.append((stream, line))
                except GeneratorExit:
                    results.append('finish')
                    raise
        yield self.do_test_sequence(consumer)
        self.assertEqual(results, [('o', 'hello'), ('e', 'cruel'), ('o', 'multi'), ('o', 'line'), ('o', 'chunk'), ('h', 'H1'), ('h', 'H2'), 'finish'])

    @defer.inlineCallbacks
    def test_sequence_no_finish(self):
        if False:
            for i in range(10):
                print('nop')
        results = []

        def consumer():
            if False:
                i = 10
                return i + 15
            while True:
                (stream, line) = (yield)
                results.append((stream, line))
        yield self.do_test_sequence(consumer)
        self.assertEqual(results, [('o', 'hello'), ('e', 'cruel'), ('o', 'multi'), ('o', 'line'), ('o', 'chunk'), ('h', 'H1'), ('h', 'H2')])

class TestLogLineObserver(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)

    @defer.inlineCallbacks
    def test_sequence(self):
        if False:
            i = 10
            return i + 15
        logid = (yield self.master.data.updates.addLog(1, 'mine', 's'))
        _log = log.Log.new(self.master, 'mine', 's', logid, 'utf-8')
        lo = MyLogLineObserver()
        lo.setLog(_log)
        yield _log.addStdout('hello\n')
        yield _log.addStderr('cruel\n')
        yield _log.addStdout('multi\nline\nchunk\n')
        yield _log.addHeader('H1\nH2\n')
        yield _log.finish()
        self.assertEqual(lo.obs, [('out', 'hello'), ('err', 'cruel'), ('out', 'multi'), ('out', 'line'), ('out', 'chunk'), ('hdr', 'H1'), ('hdr', 'H2'), ('fin',)])

    def test_old_setMaxLineLength(self):
        if False:
            for i in range(10):
                print('nop')
        lo = MyLogLineObserver()
        lo.setMaxLineLength(120939403)

class TestOutputProgressObserver(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)

    @defer.inlineCallbacks
    def test_sequence(self):
        if False:
            return 10
        logid = (yield self.master.data.updates.addLog(1, 'mine', 's'))
        _log = log.Log.new(self.master, 'mine', 's', logid, 'utf-8')
        lo = logobserver.OutputProgressObserver('stdio')
        step = mock.Mock()
        lo.setStep(step)
        lo.setLog(_log)
        yield _log.addStdout('hello\n')
        step.setProgress.assert_called_with('stdio', 6)
        yield _log.finish()

class TestBufferObserver(TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)

    @defer.inlineCallbacks
    def do_test_sequence(self, lo):
        if False:
            i = 10
            return i + 15
        logid = (yield self.master.data.updates.addLog(1, 'mine', 's'))
        _log = log.Log.new(self.master, 'mine', 's', logid, 'utf-8')
        lo.setLog(_log)
        yield _log.addStdout('hello\n')
        yield _log.addStderr('cruel\n')
        yield _log.addStdout('multi\nline\nchunk\n')
        yield _log.addHeader('H1\nH2\n')
        yield _log.finish()

    @defer.inlineCallbacks
    def test_stdout_only(self):
        if False:
            print('Hello World!')
        lo = logobserver.BufferLogObserver(wantStdout=True, wantStderr=False)
        yield self.do_test_sequence(lo)
        self.assertEqual(lo.getStdout(), 'hello\nmulti\nline\nchunk\n')
        self.assertEqual(lo.getStderr(), '')

    @defer.inlineCallbacks
    def test_both(self):
        if False:
            return 10
        lo = logobserver.BufferLogObserver(wantStdout=True, wantStderr=True)
        yield self.do_test_sequence(lo)
        self.assertEqual(lo.getStdout(), 'hello\nmulti\nline\nchunk\n')
        self.assertEqual(lo.getStderr(), 'cruel\n')
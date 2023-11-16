"""
Tests for L{twisted.runner.procmon}.
"""
import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest

class DummyProcess:
    """
    An incomplete and fake L{IProcessTransport} implementation for testing how
    L{ProcessMonitor} behaves when its monitored processes exit.

    @ivar _terminationDelay: the delay in seconds after which the DummyProcess
        will appear to exit when it receives a TERM signal
    """
    pid = 1
    proto = None
    _terminationDelay = 1

    def __init__(self, reactor, executable, args, environment, path, proto, uid=None, gid=None, usePTY=0, childFDs=None):
        if False:
            for i in range(10):
                print('nop')
        self.proto = proto
        self._reactor = reactor
        self._executable = executable
        self._args = args
        self._environment = environment
        self._path = path
        self._uid = uid
        self._gid = gid
        self._usePTY = usePTY
        self._childFDs = childFDs

    def signalProcess(self, signalID):
        if False:
            i = 10
            return i + 15
        '\n        A partial implementation of signalProcess which can only handle TERM and\n        KILL signals.\n         - When a TERM signal is given, the dummy process will appear to exit\n           after L{DummyProcess._terminationDelay} seconds with exit code 0\n         - When a KILL signal is given, the dummy process will appear to exit\n           immediately with exit code 1.\n\n        @param signalID: The signal name or number to be issued to the process.\n        @type signalID: C{str}\n        '
        params = {'TERM': (self._terminationDelay, 0), 'KILL': (0, 1)}
        if self.pid is None:
            raise ProcessExitedAlready()
        if signalID in params:
            (delay, status) = params[signalID]
            self._signalHandler = self._reactor.callLater(delay, self.processEnded, status)

    def processEnded(self, status):
        if False:
            return 10
        '\n        Deliver the process ended event to C{self.proto}.\n        '
        self.pid = None
        statusMap = {0: ProcessDone, 1: ProcessTerminated}
        self.proto.processEnded(Failure(statusMap[status](status)))

class DummyProcessReactor(MemoryReactor, Clock):
    """
    @ivar spawnedProcesses: a list that keeps track of the fake process
        instances built by C{spawnProcess}.
    @type spawnedProcesses: C{list}
    """

    def __init__(self):
        if False:
            return 10
        MemoryReactor.__init__(self)
        Clock.__init__(self)
        self.spawnedProcesses = []

    def spawnProcess(self, processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        if False:
            while True:
                i = 10
        '\n        Fake L{reactor.spawnProcess}, that logs all the process\n        arguments and returns a L{DummyProcess}.\n        '
        proc = DummyProcess(self, executable, args, env, path, processProtocol, uid, gid, usePTY, childFDs)
        processProtocol.makeConnection(proc)
        self.spawnedProcesses.append(proc)
        return proc

class ProcmonTests(unittest.TestCase):
    """
    Tests for L{ProcessMonitor}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an L{ProcessMonitor} wrapped around a fake reactor.\n        '
        self.reactor = DummyProcessReactor()
        self.pm = ProcessMonitor(reactor=self.reactor)
        self.pm.minRestartDelay = 2
        self.pm.maxRestartDelay = 10
        self.pm.threshold = 10

    def test_reprLooksGood(self):
        if False:
            return 10
        '\n        Repr includes all details\n        '
        self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
        representation = repr(self.pm)
        self.assertIn('foo', representation)
        self.assertIn('1', representation)
        self.assertIn('2', representation)

    def test_simpleReprLooksGood(self):
        if False:
            while True:
                i = 10
        '\n        Repr does not include unneeded details.\n\n        Values of attributes that just mean "inherit from launching\n        process" do not appear in the repr of a process.\n        '
        self.pm.addProcess('foo', ['arg1', 'arg2'], env={})
        representation = repr(self.pm)
        self.assertNotIn('(', representation)
        self.assertNotIn(')', representation)

    def test_getStateIncludesProcesses(self):
        if False:
            while True:
                i = 10
        '\n        The list of monitored processes must be included in the pickle state.\n        '
        self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
        self.assertEqual(self.pm.__getstate__()['processes'], {'foo': (['arg1', 'arg2'], 1, 2, {})})

    def test_getStateExcludesReactor(self):
        if False:
            print('Hello World!')
        '\n        The private L{ProcessMonitor._reactor} instance variable should not be\n        included in the pickle state.\n        '
        self.assertNotIn('_reactor', self.pm.__getstate__())

    def test_addProcess(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.addProcess} only starts the named program if\n        L{ProcessMonitor.startService} has been called.\n        '
        self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
        self.assertEqual(self.pm.protocols, {})
        self.assertEqual(self.pm.processes, {'foo': (['arg1', 'arg2'], 1, 2, {})})
        self.pm.startService()
        self.reactor.advance(0)
        self.assertEqual(list(self.pm.protocols.keys()), ['foo'])

    def test_addProcessDuplicateKeyError(self):
        if False:
            return 10
        '\n        L{ProcessMonitor.addProcess} raises a C{KeyError} if a process with the\n        given name already exists.\n        '
        self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
        self.assertRaises(KeyError, self.pm.addProcess, 'foo', ['arg1', 'arg2'], uid=1, gid=2, env={})

    def test_addProcessEnv(self):
        if False:
            while True:
                i = 10
        '\n        L{ProcessMonitor.addProcess} takes an C{env} parameter that is passed to\n        L{IReactorProcess.spawnProcess}.\n        '
        fakeEnv = {'KEY': 'value'}
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'], uid=1, gid=2, env=fakeEnv)
        self.reactor.advance(0)
        self.assertEqual(self.reactor.spawnedProcesses[0]._environment, fakeEnv)

    def test_addProcessCwd(self):
        if False:
            print('Hello World!')
        '\n        L{ProcessMonitor.addProcess} takes an C{cwd} parameter that is passed\n        to L{IReactorProcess.spawnProcess}.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'], cwd='/mnt/lala')
        self.reactor.advance(0)
        self.assertEqual(self.reactor.spawnedProcesses[0]._path, '/mnt/lala')

    def test_removeProcess(self):
        if False:
            return 10
        '\n        L{ProcessMonitor.removeProcess} removes the process from the public\n        processes list.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.assertEqual(len(self.pm.processes), 1)
        self.pm.removeProcess('foo')
        self.assertEqual(len(self.pm.processes), 0)

    def test_removeProcessUnknownKeyError(self):
        if False:
            print('Hello World!')
        "\n        L{ProcessMonitor.removeProcess} raises a C{KeyError} if the given\n        process name isn't recognised.\n        "
        self.pm.startService()
        self.assertRaises(KeyError, self.pm.removeProcess, 'foo')

    def test_startProcess(self):
        if False:
            return 10
        '\n        When a process has been started, an instance of L{LoggingProtocol} will\n        be added to the L{ProcessMonitor.protocols} dict and the start time of\n        the process will be recorded in the L{ProcessMonitor.timeStarted}\n        dictionary.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startProcess('foo')
        self.assertIsInstance(self.pm.protocols['foo'], LoggingProtocol)
        self.assertIn('foo', self.pm.timeStarted.keys())

    def test_startProcessAlreadyStarted(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.startProcess} silently returns if the named process is\n        already started.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startProcess('foo')
        self.assertIsNone(self.pm.startProcess('foo'))

    def test_startProcessUnknownKeyError(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{ProcessMonitor.startProcess} raises a C{KeyError} if the given\n        process name isn't recognised.\n        "
        self.assertRaises(KeyError, self.pm.startProcess, 'foo')

    def test_stopProcessNaturalTermination(self):
        if False:
            while True:
                i = 10
        '\n        L{ProcessMonitor.stopProcess} immediately sends a TERM signal to the\n        named process.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.assertIn('foo', self.pm.protocols)
        timeToDie = self.pm.protocols['foo'].transport._terminationDelay = 1
        self.reactor.advance(self.pm.threshold)
        self.pm.stopProcess('foo')
        self.reactor.advance(timeToDie)
        self.reactor.advance(0)
        self.assertEqual(self.reactor.seconds(), self.pm.timeStarted['foo'])

    def test_stopProcessForcedKill(self):
        if False:
            return 10
        '\n        L{ProcessMonitor.stopProcess} kills a process which fails to terminate\n        naturally within L{ProcessMonitor.killTime} seconds.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        proc = self.pm.protocols['foo'].transport
        proc._terminationDelay = self.pm.killTime + 1
        self.pm.stopProcess('foo')
        self.reactor.advance(self.pm.killTime - 1)
        self.assertEqual(0.0, self.pm.timeStarted['foo'])
        self.reactor.advance(1)
        self.reactor.pump([0, 0])
        self.assertEqual(self.reactor.seconds(), self.pm.timeStarted['foo'])

    def test_stopProcessUnknownKeyError(self):
        if False:
            i = 10
            return i + 15
        "\n        L{ProcessMonitor.stopProcess} raises a C{KeyError} if the given process\n        name isn't recognised.\n        "
        self.assertRaises(KeyError, self.pm.stopProcess, 'foo')

    def test_stopProcessAlreadyStopped(self):
        if False:
            while True:
                i = 10
        '\n        L{ProcessMonitor.stopProcess} silently returns if the named process\n        is already stopped. eg Process has crashed and a restart has been\n        rescheduled, but in the meantime, the service is stopped.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.assertIsNone(self.pm.stopProcess('foo'))

    def test_outputReceivedCompleteLine(self):
        if False:
            i = 10
            return i + 15
        '\n        Getting a complete output line on stdout generates a log message.\n        '
        events = []
        self.addCleanup(globalLogPublisher.removeObserver, events.append)
        globalLogPublisher.addObserver(events.append)
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        self.pm.protocols['foo'].outReceived(b'hello world!\n')
        self.assertEquals(len(events), 1)
        namespace = events[0]['log_namespace']
        stream = events[0]['stream']
        tag = events[0]['tag']
        line = events[0]['line']
        self.assertEquals(namespace, 'twisted.runner.procmon.ProcessMonitor')
        self.assertEquals(stream, 'stdout')
        self.assertEquals(tag, 'foo')
        self.assertEquals(line, 'hello world!')

    def test_ouputReceivedCompleteErrLine(self):
        if False:
            i = 10
            return i + 15
        '\n        Getting a complete output line on stderr generates a log message.\n        '
        events = []
        self.addCleanup(globalLogPublisher.removeObserver, events.append)
        globalLogPublisher.addObserver(events.append)
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        self.pm.protocols['foo'].errReceived(b'hello world!\n')
        self.assertEquals(len(events), 1)
        namespace = events[0]['log_namespace']
        stream = events[0]['stream']
        tag = events[0]['tag']
        line = events[0]['line']
        self.assertEquals(namespace, 'twisted.runner.procmon.ProcessMonitor')
        self.assertEquals(stream, 'stderr')
        self.assertEquals(tag, 'foo')
        self.assertEquals(line, 'hello world!')

    def test_outputReceivedCompleteLineInvalidUTF8(self):
        if False:
            return 10
        '\n        Getting invalid UTF-8 results in the repr of the raw message\n        '
        events = []
        self.addCleanup(globalLogPublisher.removeObserver, events.append)
        globalLogPublisher.addObserver(events.append)
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        self.pm.protocols['foo'].outReceived(b'\xffhello world!\n')
        self.assertEquals(len(events), 1)
        message = events[0]
        namespace = message['log_namespace']
        stream = message['stream']
        tag = message['tag']
        output = message['line']
        self.assertEquals(namespace, 'twisted.runner.procmon.ProcessMonitor')
        self.assertEquals(stream, 'stdout')
        self.assertEquals(tag, 'foo')
        self.assertEquals(output, repr(b'\xffhello world!'))

    def test_outputReceivedPartialLine(self):
        if False:
            return 10
        '\n        Getting partial line results in no events until process end\n        '
        events = []
        self.addCleanup(globalLogPublisher.removeObserver, events.append)
        globalLogPublisher.addObserver(events.append)
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        self.pm.protocols['foo'].outReceived(b'hello world!')
        self.assertEquals(len(events), 0)
        self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
        self.assertEquals(len(events), 1)
        namespace = events[0]['log_namespace']
        stream = events[0]['stream']
        tag = events[0]['tag']
        line = events[0]['line']
        self.assertEquals(namespace, 'twisted.runner.procmon.ProcessMonitor')
        self.assertEquals(stream, 'stdout')
        self.assertEquals(tag, 'foo')
        self.assertEquals(line, 'hello world!')

    def test_connectionLostLongLivedProcess(self):
        if False:
            print('Hello World!')
        '\n        L{ProcessMonitor.connectionLost} should immediately restart a process\n        if it has been running longer than L{ProcessMonitor.threshold} seconds.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(self.pm.threshold)
        self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
        self.assertNotIn('foo', self.pm.protocols)
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)

    def test_connectionLostMurderCancel(self):
        if False:
            print('Hello World!')
        '\n        L{ProcessMonitor.connectionLost} cancels a scheduled process killer and\n        deletes the DelayedCall from the L{ProcessMonitor.murder} list.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(1)
        self.pm.stopProcess('foo')
        self.assertIn('foo', self.pm.murder)
        delayedCall = self.pm.murder['foo']
        self.assertTrue(delayedCall.active())
        self.reactor.advance(self.pm.protocols['foo'].transport._terminationDelay)
        self.assertFalse(delayedCall.active())
        self.assertNotIn('foo', self.pm.murder)

    def test_connectionLostProtocolDeletion(self):
        if False:
            while True:
                i = 10
        '\n        L{ProcessMonitor.connectionLost} removes the corresponding\n        ProcessProtocol instance from the L{ProcessMonitor.protocols} list.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.assertIn('foo', self.pm.protocols)
        self.pm.protocols['foo'].transport.signalProcess('KILL')
        self.reactor.advance(self.pm.protocols['foo'].transport._terminationDelay)
        self.assertNotIn('foo', self.pm.protocols)

    def test_connectionLostMinMaxRestartDelay(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.connectionLost} will wait at least minRestartDelay s\n        and at most maxRestartDelay s\n        '
        self.pm.minRestartDelay = 2
        self.pm.maxRestartDelay = 3
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.assertEqual(self.pm.delay['foo'], self.pm.minRestartDelay)
        self.reactor.advance(self.pm.threshold - 1)
        self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
        self.assertEqual(self.pm.delay['foo'], self.pm.maxRestartDelay)

    def test_connectionLostBackoffDelayDoubles(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.connectionLost} doubles the restart delay each time\n        the process dies too quickly.\n        '
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.reactor.advance(self.pm.threshold - 1)
        self.assertIn('foo', self.pm.protocols)
        self.assertEqual(self.pm.delay['foo'], self.pm.minRestartDelay)
        self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
        self.assertEqual(self.pm.delay['foo'], self.pm.minRestartDelay * 2)

    def test_startService(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.startService} starts all monitored processes.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(0)
        self.assertIn('foo', self.pm.protocols)

    def test_stopService(self):
        if False:
            return 10
        '\n        L{ProcessMonitor.stopService} should stop all monitored processes.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.addProcess('bar', ['bar'])
        self.pm.startService()
        self.reactor.advance(self.pm.threshold)
        self.assertIn('foo', self.pm.protocols)
        self.assertIn('bar', self.pm.protocols)
        self.reactor.advance(1)
        self.pm.stopService()
        self.reactor.advance(self.pm.killTime + 1)
        self.assertEqual({}, self.pm.protocols)

    def test_restartAllRestartsOneProcess(self):
        if False:
            while True:
                i = 10
        '\n        L{ProcessMonitor.restartAll} succeeds when there is one process.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(1)
        self.pm.restartAll()
        self.reactor.advance(1)
        processes = list(self.reactor.spawnedProcesses)
        myProcess = processes.pop()
        self.assertEquals(processes, [])
        self.assertIsNone(myProcess.pid)

    def test_stopServiceCancelRestarts(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ProcessMonitor.stopService} should cancel any scheduled process\n        restarts.\n        '
        self.pm.addProcess('foo', ['foo'])
        self.pm.startService()
        self.reactor.advance(self.pm.threshold)
        self.assertIn('foo', self.pm.protocols)
        self.reactor.advance(1)
        self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
        self.assertTrue(self.pm.restart['foo'].active())
        self.pm.stopService()
        self.assertFalse(self.pm.restart['foo'].active())

    def test_stopServiceCleanupScheduledRestarts(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ProcessMonitor.stopService} should cancel all scheduled process\n        restarts.\n        '
        self.pm.threshold = 5
        self.pm.minRestartDelay = 5
        self.pm.startService()
        self.pm.addProcess('foo', ['foo'])
        self.reactor.advance(1)
        self.pm.stopProcess('foo')
        self.reactor.advance(1)
        self.pm.stopService()
        self.reactor.advance(6)
        self.assertEqual(self.pm.protocols, {})

class DeprecationTests(unittest.SynchronousTestCase):
    """
    Tests that check functionality that should be deprecated is deprecated.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create reactor and process monitor.\n        '
        self.reactor = DummyProcessReactor()
        self.pm = ProcessMonitor(reactor=self.reactor)

    def test_toTuple(self):
        if False:
            while True:
                i = 10
        '\n        _Process.toTuple is deprecated.\n\n        When getting the deprecated processes property, the actual\n        data (kept in the class _Process) is converted to a tuple --\n        which produces a DeprecationWarning per process so converted.\n        '
        self.pm.addProcess('foo', ['foo'])
        myprocesses = self.pm.processes
        self.assertEquals(len(myprocesses), 1)
        warnings = self.flushWarnings()
        foundToTuple = False
        for warning in warnings:
            self.assertIs(warning['category'], DeprecationWarning)
            if 'toTuple' in warning['message']:
                foundToTuple = True
        self.assertTrue(foundToTuple, f'no tuple deprecation found:{repr(warnings)}')

    def test_processes(self):
        if False:
            i = 10
            return i + 15
        '\n        Accessing L{ProcessMonitor.processes} results in deprecation warning\n\n        Even when there are no processes, and thus no process is converted\n        to a tuple, accessing the L{ProcessMonitor.processes} property\n        should generate its own DeprecationWarning.\n        '
        myProcesses = self.pm.processes
        self.assertEquals(myProcesses, {})
        warnings = self.flushWarnings()
        first = warnings.pop(0)
        self.assertIs(first['category'], DeprecationWarning)
        self.assertEquals(warnings, [])

    def test_getstate(self):
        if False:
            while True:
                i = 10
        '\n        Pickling an L{ProcessMonitor} results in deprecation warnings\n        '
        pickle.dumps(self.pm)
        warnings = self.flushWarnings()
        for warning in warnings:
            self.assertIs(warning['category'], DeprecationWarning)
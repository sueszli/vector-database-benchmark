"""
Tests for L{twisted.trial._dist.disttrial}.
"""
import os
import sys
from functools import partial
from io import StringIO
from os.path import sep
from typing import Callable, List, Set
from unittest import TestCase as PyUnitTestCase
from zope.interface import implementer, verify
from attrs import Factory, assoc, define, field
from hamcrest import assert_that, contains, ends_with, equal_to, has_length, none, starts_with
from hamcrest.core.core.allof import AllOf
from hypothesis import given
from hypothesis.strategies import booleans, sampled_from
from twisted.internet import interfaces
from twisted.internet.base import ReactorBase
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol, Protocol
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.trial._dist import _WORKER_AMP_STDIN
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial._dist.disttrial import DistTrialRunner, WorkerPool, WorkerPoolConfig
from twisted.trial._dist.functional import countingCalls, discardResult, fromOptional, iterateWhile, sequence
from twisted.trial._dist.worker import LocalWorker, RunResult, Worker, WorkerAction
from twisted.trial.reporter import Reporter, TestResult, TreeReporter, UncleanWarningsReporterWrapper
from twisted.trial.runner import ErrorHolder, TrialSuite
from twisted.trial.unittest import SynchronousTestCase, TestCase
from ...test import erroneous, sample
from .matchers import matches_result

@define
class FakeTransport:
    """
    A simple fake process transport.
    """
    _closed: Set[int] = field(default=Factory(set))

    def writeToChild(self, fd, data):
        if False:
            while True:
                i = 10
        '\n        Ignore write calls.\n        '

    def closeChildFD(self, fd):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mark one of the child descriptors as closed.\n        '
        self._closed.add(fd)

@implementer(interfaces.IReactorProcess)
class CountingReactor(MemoryReactorClock):
    """
    A fake reactor that counts the calls to L{IReactorCore.run},
    L{IReactorCore.stop}, and L{IReactorProcess.spawnProcess}.
    """
    spawnCount = 0
    stopCount = 0
    runCount = 0

    def __init__(self, workers):
        if False:
            for i in range(10):
                print('nop')
        MemoryReactorClock.__init__(self)
        self._workers = workers

    def spawnProcess(self, workerProto, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        See L{IReactorProcess.spawnProcess}.\n\n        @param workerProto: See L{IReactorProcess.spawnProcess}.\n        @param args: See L{IReactorProcess.spawnProcess}.\n        @param kwargs: See L{IReactorProcess.spawnProcess}.\n        '
        self._workers.append(workerProto)
        workerProto.makeConnection(FakeTransport())
        self.spawnCount += 1

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        See L{IReactorCore.stop}.\n        '
        MemoryReactorClock.stop(self)
        if 'before' in self.triggers:
            self.triggers['before']['shutdown'][0][0]()
        self.stopCount += 1

    def run(self):
        if False:
            print('Hello World!')
        '\n        See L{IReactorCore.run}.\n        '
        self.runCount += 1
        self.running = True
        self.hasRun = True
        for (f, args, kwargs) in self.whenRunningHooks:
            f(*args, **kwargs)
        self.stop()
        self.stopCount -= 1

class CountingReactorTests(SynchronousTestCase):
    """
    Tests for L{CountingReactor}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.workers = []
        self.reactor = CountingReactor(self.workers)

    def test_providesIReactorProcess(self):
        if False:
            while True:
                i = 10
        '\n        L{CountingReactor} instances provide L{IReactorProcess}.\n        '
        verify.verifyObject(interfaces.IReactorProcess, self.reactor)

    def test_spawnProcess(self):
        if False:
            while True:
                i = 10
        "\n        The process protocol for a spawned process is connected to a\n        transport and appended onto the provided C{workers} list, and\n        the reactor's C{spawnCount} increased.\n        "
        self.assertFalse(self.reactor.spawnCount)
        proto = Protocol()
        for count in [1, 2]:
            self.reactor.spawnProcess(proto, sys.executable, args=[sys.executable])
            self.assertTrue(proto.transport)
            self.assertEqual(self.workers, [proto] * count)
            self.assertEqual(self.reactor.spawnCount, count)

    def test_stop(self):
        if False:
            return 10
        '\n        Stopping the reactor increments its C{stopCount}\n        '
        self.assertFalse(self.reactor.stopCount)
        for count in [1, 2]:
            self.reactor.stop()
            self.assertEqual(self.reactor.stopCount, count)

    def test_run(self):
        if False:
            return 10
        '\n        Running the reactor increments its C{runCount}, does not imply\n        C{stop}, and calls L{IReactorCore.callWhenRunning} hooks.\n        '
        self.assertFalse(self.reactor.runCount)
        whenRunningCalls = []
        self.reactor.callWhenRunning(whenRunningCalls.append, None)
        for count in [1, 2]:
            self.reactor.run()
            self.assertEqual(self.reactor.runCount, count)
            self.assertEqual(self.reactor.stopCount, 0)
            self.assertEqual(len(whenRunningCalls), count)

class WorkerPoolTests(TestCase):
    """
    Tests for L{WorkerPool}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.parent = FilePath(self.mktemp())
        self.workingDirectory = self.parent.child('_trial_temp')
        self.config = WorkerPoolConfig(numWorkers=4, workingDirectory=self.workingDirectory, workerArguments=[], logFile='out.log')
        self.pool = WorkerPool(self.config)

    def test_createLocalWorkers(self):
        if False:
            return 10
        '\n        C{_createLocalWorkers} iterates the list of protocols and create one\n        L{LocalWorker} for each.\n        '
        protocols = [object() for x in range(4)]
        workers = self.pool._createLocalWorkers(protocols, FilePath('path'), StringIO())
        for s in workers:
            self.assertIsInstance(s, LocalWorker)
        self.assertEqual(4, len(workers))

    def test_launchWorkerProcesses(self):
        if False:
            while True:
                i = 10
        '\n        Given a C{spawnProcess} function, C{_launchWorkerProcess} launches a\n        python process with an existing path as its argument.\n        '
        protocols = [ProcessProtocol() for i in range(4)]
        arguments = []
        environment = {}

        def fakeSpawnProcess(processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
            if False:
                while True:
                    i = 10
            arguments.append(executable)
            arguments.extend(args)
            environment.update(env)
        self.pool._launchWorkerProcesses(fakeSpawnProcess, protocols, ['foo'])
        self.assertEqual(arguments[0], arguments[1])
        self.assertTrue(os.path.exists(arguments[2]))
        self.assertEqual('foo', arguments[3])
        self.assertEqual(os.pathsep.join(sys.path), environment['PYTHONPATH'])

    def test_run(self):
        if False:
            return 10
        '\n        C{run} dispatches the given action to each of its workers exactly once.\n        '
        self.parent.makedirs()
        workers = []
        starting = self.pool.start(CountingReactor([]))
        started = self.successResultOf(starting)
        running = started.run(lambda w: succeed(workers.append(w)))
        self.successResultOf(running)
        assert_that(workers, has_length(self.config.numWorkers))

    def test_runUsedDirectory(self):
        if False:
            return 10
        '\n        L{WorkerPool.start} checks if the test directory is already locked, and if\n        it is generates a name based on it.\n        '
        self.parent.makedirs()
        lock = FilesystemLock(self.workingDirectory.path + '.lock')
        self.assertTrue(lock.lock())
        self.addCleanup(lock.unlock)
        fakeReactor = CountingReactor([])
        started = self.successResultOf(self.pool.start(fakeReactor))
        self.assertEqual(started.workingDirectory, self.workingDirectory.sibling('_trial_temp-1'))

    def test_join(self):
        if False:
            while True:
                i = 10
        '\n        L{StartedWorkerPool.join} causes all of the workers to exit, closes the\n        log file, and unlocks the test directory.\n        '
        self.parent.makedirs()
        reactor = CountingReactor([])
        started = self.successResultOf(self.pool.start(reactor))
        joining = Deferred.fromCoroutine(started.join())
        self.assertNoResult(joining)
        for w in reactor._workers:
            assert_that(w.transport._closed, contains(_WORKER_AMP_STDIN))
            for fd in w.transport._closed:
                w.childConnectionLost(fd)
            for f in [w.processExited, w.processEnded]:
                f(Failure(ProcessDone(0)))
        assert_that(self.successResultOf(joining), none())
        assert_that(started.testLog.closed, equal_to(True))
        assert_that(started.testDirLock.locked, equal_to(False))

    @given(booleans(), sampled_from(['out.log', f'subdir{sep}out.log']))
    def test_logFile(self, absolute: bool, logFile: str) -> None:
        if False:
            return 10
        '\n        L{WorkerPool.start} creates a L{StartedWorkerPool} configured with a\n        log file based on the L{WorkerPoolConfig.logFile}.\n        '
        if absolute:
            logFile = self.parent.path + sep + logFile
        config = assoc(self.config, logFile=logFile)
        if absolute:
            matches = equal_to(logFile)
        else:
            matches = AllOf(starts_with(config.workingDirectory.path), ends_with(sep + logFile))
        pool = WorkerPool(config)
        started = self.successResultOf(pool.start(CountingReactor([])))
        assert_that(started.testLog.name, matches)

class DistTrialRunnerTests(TestCase):
    """
    Tests for L{DistTrialRunner}.
    """
    suite = TrialSuite([sample.FooTest('test_foo')])

    def getRunner(self, **overrides):
        if False:
            i = 10
            return i + 15
        '\n        Create a runner for testing.\n        '
        args = dict(reporterFactory=TreeReporter, workingDirectory=self.mktemp(), stream=StringIO(), maxWorkers=4, workerArguments=[], workerPoolFactory=partial(LocalWorkerPool, autostop=True), reactor=CountingReactor([]))
        args.update(overrides)
        return DistTrialRunner(**args)

    def test_writeResults(self):
        if False:
            i = 10
            return i + 15
        '\n        L{DistTrialRunner.writeResults} writes to the stream specified in the\n        init.\n        '
        stringIO = StringIO()
        result = DistReporter(Reporter(stringIO))
        runner = self.getRunner()
        runner.writeResults(result)
        self.assertTrue(stringIO.tell() > 0)

    def test_minimalWorker(self):
        if False:
            return 10
        "\n        L{DistTrialRunner.runAsync} doesn't try to start more workers than the\n        number of tests.\n        "
        pool = None

        def recordingFactory(*a, **kw):
            if False:
                while True:
                    i = 10
            nonlocal pool
            pool = LocalWorkerPool(*a, autostop=True, **kw)
            return pool
        maxWorkers = 7
        numTests = 3
        runner = self.getRunner(maxWorkers=maxWorkers, workerPoolFactory=recordingFactory)
        suite = TrialSuite([TestCase() for n in range(numTests)])
        self.successResultOf(runner.runAsync(suite))
        assert_that(pool._started[0].workers, has_length(numTests))

    def test_runUncleanWarnings(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Running with the C{unclean-warnings} option makes L{DistTrialRunner} uses\n        the L{UncleanWarningsReporterWrapper}.\n        '
        runner = self.getRunner(uncleanWarnings=True)
        d = runner.runAsync(self.suite)
        result = self.successResultOf(d)
        self.assertIsInstance(result, DistReporter)
        self.assertIsInstance(result.original, UncleanWarningsReporterWrapper)

    def test_runWithoutTest(self):
        if False:
            return 10
        '\n        L{DistTrialRunner} can run an empty test suite.\n        '
        stream = StringIO()
        runner = self.getRunner(stream=stream)
        result = self.successResultOf(runner.runAsync(TrialSuite()))
        self.assertIsInstance(result, DistReporter)
        output = stream.getvalue()
        self.assertIn('Running 0 test', output)
        self.assertIn('PASSED', output)

    def test_runWithoutTestButWithAnError(self):
        if False:
            while True:
                i = 10
        '\n        Even if there is no test, the suite can contain an error (most likely,\n        an import error): this should make the run fail, and the error should\n        be printed.\n        '
        err = ErrorHolder('an error', Failure(RuntimeError('foo bar')))
        stream = StringIO()
        runner = self.getRunner(stream=stream)
        result = self.successResultOf(runner.runAsync(err))
        self.assertIsInstance(result, DistReporter)
        output = stream.getvalue()
        self.assertIn('Running 0 test', output)
        self.assertIn('foo bar', output)
        self.assertIn('an error', output)
        self.assertIn('errors=1', output)
        self.assertIn('FAILED', output)

    def test_runUnexpectedError(self) -> None:
        if False:
            return 10
        "\n        If for some reasons we can't connect to the worker process, the error is\n        recorded in the result object.\n        "
        runner = self.getRunner(workerPoolFactory=BrokenWorkerPool)
        result = self.successResultOf(runner.runAsync(self.suite))
        errors = result.original.errors
        assert_that(errors, has_length(1))
        assert_that(errors[0][1].type, equal_to(WorkerPoolBroken))

    def test_runUnexpectedErrorCtrlC(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        If the reactor is stopped by C-c (i.e. `run` returns before the test\n        case's Deferred has been fired) we should cancel the pending test run.\n        "
        runner = self.getRunner(workerPoolFactory=LocalWorkerPool)
        with self.assertRaises(CancelledError):
            runner.run(self.suite)

    def test_runUnexpectedWorkerError(self) -> None:
        if False:
            while True:
                i = 10
        '\n        If for some reason the worker process cannot run a test, the error is\n        recorded in the result object.\n        '
        runner = self.getRunner(workerPoolFactory=partial(LocalWorkerPool, workerFactory=_BrokenLocalWorker, autostop=True))
        result = self.successResultOf(runner.runAsync(self.suite))
        errors = result.original.errors
        assert_that(errors, has_length(1))
        assert_that(errors[0][1].type, equal_to(WorkerBroken))

    def test_runWaitForProcessesDeferreds(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{DistTrialRunner} waits for the worker pool to stop.\n        '
        pool = None

        def recordingFactory(*a, **kw):
            if False:
                return 10
            nonlocal pool
            pool = LocalWorkerPool(*a, autostop=False, **kw)
            return pool
        runner = self.getRunner(workerPoolFactory=recordingFactory)
        d = Deferred.fromCoroutine(runner.runAsync(self.suite))
        if pool is None:
            self.fail('worker pool was never created')
        assert pool is not None
        stopped = pool._started[0]._stopped
        self.assertNoResult(d)
        stopped.callback(None)
        result = self.successResultOf(d)
        self.assertIsInstance(result, DistReporter)

    def test_exitFirst(self):
        if False:
            while True:
                i = 10
        '\n        L{DistTrialRunner} can run in C{exitFirst} mode where it will run until a\n        test fails and then abandon the rest of the suite.\n        '
        stream = StringIO()
        suite = TrialSuite([sample.FooTest('test_foo'), erroneous.TestRegularFail('test_fail'), sample.FooTest('test_bar')])
        runner = self.getRunner(stream=stream, exitFirst=True, maxWorkers=2)
        d = runner.runAsync(suite)
        result = self.successResultOf(d)
        assert_that(result.original, matches_result(successes=1, failures=has_length(1)))

    def test_runUntilFailure(self):
        if False:
            print('Hello World!')
        '\n        L{DistTrialRunner} can run in C{untilFailure} mode where it will run\n        the given tests until they fail.\n        '
        stream = StringIO()
        case = erroneous.EventuallyFailingTestCase('test_it')
        runner = self.getRunner(stream=stream)
        d = runner.runAsync(case, untilFailure=True)
        result = self.successResultOf(d)
        self.assertEqual(5, case.n)
        self.assertFalse(result.wasSuccessful())
        output = stream.getvalue()
        self.assertEqual(output.count('PASSED'), case.n - 1, 'expected to see PASSED in output')
        self.assertIn('FAIL', output)
        for i in range(1, 6):
            self.assertIn(f'Test Pass {i}', output)
        self.assertEqual(output.count('Ran 1 tests in'), case.n, 'expected to see per-iteration test count in output')

    def test_run(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{DistTrialRunner.run} returns a L{DistReporter} containing the result of\n        the test suite run.\n        '
        runner = self.getRunner()
        result = runner.run(self.suite)
        assert_that(result.wasSuccessful(), equal_to(True))
        assert_that(result.successes, equal_to(1))

    def test_installedReactor(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{DistTrialRunner.run} uses the installed reactor L{DistTrialRunner} was\n        constructed without a reactor.\n        '
        reactor = CountingReactor([])
        with AlternateReactor(reactor):
            runner = self.getRunner(reactor=None)
        result = runner.run(self.suite)
        assert_that(result.errors, equal_to([]))
        assert_that(result.failures, equal_to([]))
        assert_that(result.wasSuccessful(), equal_to(True))
        assert_that(result.successes, equal_to(1))
        assert_that(reactor.runCount, equal_to(1))
        assert_that(reactor.stopCount, equal_to(1))

    def test_wrongInstalledReactor(self) -> None:
        if False:
            return 10
        '\n        L{DistTrialRunner} raises L{TypeError} if the installed reactor provides\n        neither L{IReactorCore} nor L{IReactorProcess} and no other reactor is\n        given.\n        '

        class Core(ReactorBase):

            def installWaker(self):
                if False:
                    while True:
                        i = 10
                pass

        @implementer(interfaces.IReactorProcess)
        class Process:

            def spawnProcess(self, processProtocol, executable, args, env=None, path=None, uid=None, gid=None, usePTY=False, childFDs=None):
                if False:
                    while True:
                        i = 10
                pass

        class Neither:
            pass
        with AlternateReactor(Neither()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)
        with AlternateReactor(Core()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)
        with AlternateReactor(Process()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)

    def test_runFailure(self):
        if False:
            i = 10
            return i + 15
        '\n        If there is an unexpected exception running the test suite then it is\n        re-raised by L{DistTrialRunner.run}.\n        '

        class BrokenFactory(Exception):
            pass

        def brokenFactory(*args, **kwargs):
            if False:
                print('Hello World!')
            raise BrokenFactory()
        runner = self.getRunner(workerPoolFactory=brokenFactory)
        with self.assertRaises(BrokenFactory):
            runner.run(self.suite)

class FunctionalTests(TestCase):
    """
    Tests for the functional helpers that need it.
    """

    def test_fromOptional(self) -> None:
        if False:
            return 10
        '\n        ``fromOptional`` accepts a default value and an ``Optional`` value of the\n        same type and returns the default value if the optional value is\n        ``None`` or the optional value otherwise.\n        '
        assert_that(fromOptional(1, None), equal_to(1))
        assert_that(fromOptional(2, 2), equal_to(2))

    def test_discardResult(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        ``discardResult`` accepts an awaitable and returns a ``Deferred`` that\n        fires with ``None`` after the awaitable completes.\n        '
        a: Deferred[str] = Deferred()
        d = discardResult(a)
        self.assertNoResult(d)
        a.callback('result')
        assert_that(self.successResultOf(d), none())

    def test_sequence(self):
        if False:
            return 10
        '\n        ``sequence`` accepts two awaitables and returns an awaitable that waits\n        for the first one to complete and then completes with the result of\n        the second one.\n        '
        a: Deferred[str] = Deferred()
        b: Deferred[int] = Deferred()
        c = Deferred.fromCoroutine(sequence(a, b))
        b.callback(42)
        self.assertNoResult(c)
        a.callback('hello')
        assert_that(self.successResultOf(c), equal_to(42))

    def test_iterateWhile(self):
        if False:
            print('Hello World!')
        '\n        ``iterateWhile`` executes the actions from its factory until the predicate\n        does not match an action result.\n        '
        actions: List[Deferred[int]] = [Deferred(), Deferred(), Deferred()]

        def predicate(value):
            if False:
                print('Hello World!')
            return value != 42
        d: Deferred[int] = Deferred.fromCoroutine(iterateWhile(predicate, list(actions).pop))
        actions.pop().callback(7)
        self.assertNoResult(d)
        actions.pop().callback(42)
        assert_that(self.successResultOf(d), equal_to(42))

    def test_countingCalls(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        ``countingCalls`` decorates a function so that it is called with an\n        increasing counter and passes the return value through.\n        '

        @countingCalls
        def target(n: int) -> int:
            if False:
                i = 10
                return i + 15
            return n + 1
        for expected in range(1, 10):
            assert_that(target(), equal_to(expected))

class WorkerPoolBroken(Exception):
    """
    An exception for ``StartedWorkerPoolBroken`` to fail with to allow tests
    to exercise exception code paths.
    """

class StartedWorkerPoolBroken:
    """
    A broken, started worker pool.  Its workers cannot run actions.  They
    always raise an exception.
    """

    async def run(self, workerAction: WorkerAction[None]) -> None:
        raise WorkerPoolBroken()

    async def join(self) -> None:
        return None

@define
class BrokenWorkerPool:
    """
    A worker pool that has workers with a broken ``run`` method.
    """
    _config: WorkerPoolConfig

    async def start(self, reactor: interfaces.IReactorProcess) -> StartedWorkerPoolBroken:
        return StartedWorkerPoolBroken()

class _LocalWorker:
    """
    A L{Worker} that runs tests in this process in the usual way.

    This is a test double for L{LocalWorkerAMP} which allows testing worker
    pool logic without sending tests over an AMP connection to be run
    somewhere else..
    """

    async def run(self, case: PyUnitTestCase, result: TestResult) -> RunResult:
        """
        Directly run C{case} in the usual way.
        """
        TrialSuite([case]).run(result)
        return {'success': True}

class WorkerBroken(Exception):
    """
    A worker tried to run a test case but the worker is broken.
    """

class _BrokenLocalWorker:
    """
    A L{Worker} that always fails to run test cases.
    """

    async def run(self, case: PyUnitTestCase, result: TestResult) -> None:
        """
        Raise an exception instead of running C{case}.
        """
        raise WorkerBroken()

@define
class StartedLocalWorkerPool:
    """
    A started L{LocalWorkerPool}.
    """
    workingDirectory: FilePath[str]
    workers: List[Worker]
    _stopped: Deferred[None]

    async def run(self, workerAction: WorkerAction[None]) -> None:
        """
        Run the action with each local worker.
        """
        for worker in self.workers:
            await workerAction(worker)

    async def join(self):
        await self._stopped

@define
class LocalWorkerPool:
    """
    Implement a worker pool that runs tests in-process instead of in child
    processes.
    """
    _config: WorkerPoolConfig
    _started: List[StartedLocalWorkerPool] = field(default=Factory(list))
    _autostop: bool = False
    _workerFactory: Callable[[], Worker] = _LocalWorker

    async def start(self, reactor: interfaces.IReactorProcess) -> StartedLocalWorkerPool:
        workers = [self._workerFactory() for i in range(self._config.numWorkers)]
        started = StartedLocalWorkerPool(self._config.workingDirectory, workers, succeed(None) if self._autostop else Deferred())
        self._started.append(started)
        return started
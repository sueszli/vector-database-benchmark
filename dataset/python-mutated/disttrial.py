"""
This module contains the trial distributed runner, the management class
responsible for coordinating all of trial's behavior at the highest level.

@since: 12.3
"""
import os
import sys
from functools import partial
from os.path import isabs
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence, TextIO, Union, cast
from unittest import TestCase, TestSuite
from attrs import define, field, frozen
from attrs.converters import default_if_none
from twisted.internet.defer import Deferred, DeferredList, gatherResults
from twisted.internet.interfaces import IReactorCore, IReactorProcess
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.python.modules import theSystemPath
from .._asyncrunner import _iterateTests
from ..itrial import IReporter, ITestCase
from ..reporter import UncleanWarningsReporterWrapper
from ..runner import TestHolder
from ..util import _unusedTestDirectory, openTestLog
from . import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT
from .distreporter import DistReporter
from .functional import countingCalls, discardResult, iterateWhile, takeWhile
from .worker import LocalWorker, LocalWorkerAMP, WorkerAction

class IDistTrialReactor(IReactorCore, IReactorProcess):
    """
    The reactor interfaces required by disttrial.
    """

def _defaultReactor() -> IDistTrialReactor:
    if False:
        return 10
    '\n    Get the default reactor, ensuring it is suitable for use with disttrial.\n    '
    import twisted.internet.reactor as defaultReactor
    if all([IReactorCore.providedBy(defaultReactor), IReactorProcess.providedBy(defaultReactor)]):
        return cast(IDistTrialReactor, defaultReactor)
    raise TypeError('Reactor does not provide the right interfaces')

@frozen
class WorkerPoolConfig:
    """
    Configuration parameters for a pool of test-running workers.

    @ivar numWorkers: The number of workers in the pool.

    @ivar workingDirectory: A directory in which working directories for each
        of the workers will be created.

    @ivar workerArguments: Extra arguments to pass the worker process in its
        argv.

    @ivar logFile: The basename of the overall test log file.
    """
    numWorkers: int
    workingDirectory: FilePath[Any]
    workerArguments: Sequence[str]
    logFile: str

@define
class StartedWorkerPool:
    """
    A pool of workers which have already been started.

    @ivar workingDirectory: A directory holding the working directories for
        each of the workers.

    @ivar testDirLock: An object representing the cooperative lock this pool
        holds on its working directory.

    @ivar testLog: The open overall test log file.

    @ivar workers: Objects corresponding to the worker child processes and
        adapting between process-related interfaces and C{IProtocol}.

    @ivar ampWorkers: AMP protocol instances corresponding to the worker child
        processes.
    """
    workingDirectory: FilePath[Any]
    testDirLock: FilesystemLock
    testLog: TextIO
    workers: List[LocalWorker]
    ampWorkers: List[LocalWorkerAMP]
    _logger = Logger()

    async def run(self, workerAction: WorkerAction[Any]) -> None:
        """
        Run an action on all of the workers in the pool.
        """
        await gatherResults((discardResult(workerAction(worker)) for worker in self.ampWorkers))
        return None

    async def join(self) -> None:
        """
        Shut down all of the workers in the pool.

        The pool is unusable after this method is called.
        """
        results = await DeferredList([Deferred.fromCoroutine(worker.exit()) for worker in self.workers], consumeErrors=True)
        for (n, (succeeded, failure)) in enumerate(results):
            if not succeeded:
                self._logger.failure(f'joining disttrial worker #{n} failed', failure)
        del self.workers[:]
        del self.ampWorkers[:]
        self.testLog.close()
        self.testDirLock.unlock()

@frozen
class WorkerPool:
    """
    Manage a fixed-size collection of child processes which can run tests.

    @ivar _config: Configuration for the precise way in which the pool is run.
    """
    _config: WorkerPoolConfig

    def _createLocalWorkers(self, protocols: Iterable[LocalWorkerAMP], workingDirectory: FilePath[Any], logFile: TextIO) -> List[LocalWorker]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create local worker protocol instances and return them.\n\n        @param protocols: The process/protocol adapters to use for the created\n        workers.\n\n        @param workingDirectory: The base path in which we should run the\n            workers.\n\n        @param logFile: The test log, for workers to write to.\n\n        @return: A list of C{quantity} C{LocalWorker} instances.\n        '
        return [LocalWorker(protocol, workingDirectory.child(str(x)), logFile) for (x, protocol) in enumerate(protocols)]

    def _launchWorkerProcesses(self, spawner, protocols, arguments):
        if False:
            return 10
        '\n        Spawn processes from a list of process protocols.\n\n        @param spawner: A C{IReactorProcess.spawnProcess} implementation.\n\n        @param protocols: An iterable of C{ProcessProtocol} instances.\n\n        @param arguments: Extra arguments passed to the processes.\n        '
        workertrialPath = theSystemPath['twisted.trial._dist.workertrial'].filePath.path
        childFDs = {0: 'w', 1: 'r', 2: 'r', _WORKER_AMP_STDIN: 'w', _WORKER_AMP_STDOUT: 'r'}
        environ = os.environ.copy()
        environ['PYTHONPATH'] = os.pathsep.join(sys.path)
        for worker in protocols:
            args = [sys.executable, workertrialPath]
            args.extend(arguments)
            spawner(worker, sys.executable, args=args, childFDs=childFDs, env=environ)

    async def start(self, reactor: IReactorProcess) -> StartedWorkerPool:
        """
        Launch all of the workers for this pool.

        @return: A started pool object that can run jobs using the workers.
        """
        (testDir, testDirLock) = _unusedTestDirectory(self._config.workingDirectory)
        if isabs(self._config.logFile):
            testLogPath = FilePath(self._config.logFile)
        else:
            testLogPath = testDir.preauthChild(self._config.logFile)
        testLog = openTestLog(testLogPath)
        ampWorkers = [LocalWorkerAMP() for x in range(self._config.numWorkers)]
        workers = self._createLocalWorkers(ampWorkers, testDir, testLog)
        self._launchWorkerProcesses(reactor.spawnProcess, workers, self._config.workerArguments)
        return StartedWorkerPool(testDir, testDirLock, testLog, workers, ampWorkers)

def shouldContinue(untilFailure: bool, result: IReporter) -> bool:
    if False:
        print('Hello World!')
    '\n    Determine whether the test suite should be iterated again.\n\n    @param untilFailure: C{True} if the suite is supposed to run until\n        failure.\n\n    @param result: The test result of the test suite iteration which just\n        completed.\n    '
    return untilFailure and result.wasSuccessful()

async def runTests(pool: StartedWorkerPool, testCases: Iterable[ITestCase], result: DistReporter, driveWorker: Callable[[DistReporter, Sequence[ITestCase], LocalWorkerAMP], Awaitable[None]]) -> None:
    try:
        await pool.run(partial(driveWorker, result, testCases))
    except Exception:
        result.original.addError(TestHolder('<runTests>'), Failure())

@define
class DistTrialRunner:
    """
    A specialized runner for distributed trial. The runner launches a number of
    local worker processes which will run tests.

    @ivar _maxWorkers: the number of workers to be spawned.

    @ivar _exitFirst: ``True`` to stop the run as soon as a test case fails.
        ``False`` to run through the whole suite and report all of the results
        at the end.

    @ivar stream: stream which the reporter will use.

    @ivar _reporterFactory: the reporter class to be used.
    """
    _distReporterFactory = DistReporter
    _logger = Logger()
    _reporterFactory: Callable[..., IReporter]
    _maxWorkers: int
    _workerArguments: List[str]
    _exitFirst: bool = False
    _reactor: IDistTrialReactor = field(default=None, converter=default_if_none(factory=_defaultReactor))
    stream: TextIO = field(default=None, converter=default_if_none(sys.stdout))
    _tracebackFormat: str = 'default'
    _realTimeErrors: bool = False
    _uncleanWarnings: bool = False
    _logfile: str = 'test.log'
    _workingDirectory: str = '_trial_temp'
    _workerPoolFactory: Callable[[WorkerPoolConfig], WorkerPool] = WorkerPool

    def _makeResult(self) -> DistReporter:
        if False:
            print('Hello World!')
        '\n        Make reporter factory, and wrap it with a L{DistReporter}.\n        '
        reporter = self._reporterFactory(self.stream, self._tracebackFormat, realtime=self._realTimeErrors)
        if self._uncleanWarnings:
            reporter = UncleanWarningsReporterWrapper(reporter)
        return self._distReporterFactory(reporter)

    def writeResults(self, result):
        if False:
            while True:
                i = 10
        '\n        Write test run final outcome to result.\n\n        @param result: A C{TestResult} which will print errors and the summary.\n        '
        result.done()

    async def _driveWorker(self, result: DistReporter, testCases: Sequence[ITestCase], worker: LocalWorkerAMP) -> None:
        """
        Drive a L{LocalWorkerAMP} instance, iterating the tests and calling
        C{run} for every one of them.

        @param worker: The L{LocalWorkerAMP} to drive.

        @param result: The global L{DistReporter} instance.

        @param testCases: The global list of tests to iterate.

        @return: A coroutine that completes after all of the tests have
            completed.
        """

        async def task(case):
            try:
                await worker.run(case, result)
            except Exception:
                result.original.addError(case, Failure())
        for case in testCases:
            await task(case)

    async def runAsync(self, suite: Union[TestCase, TestSuite], untilFailure: bool=False) -> DistReporter:
        """
        Spawn local worker processes and load tests. After that, run them.

        @param suite: A test or suite to be run.

        @param untilFailure: If C{True}, continue to run the tests until they
            fail.

        @return: A coroutine that completes with the test result.
        """
        testCases = list(_iterateTests(suite))
        poolStarter = self._workerPoolFactory(WorkerPoolConfig(min(len(testCases), self._maxWorkers), FilePath(self._workingDirectory), self._workerArguments, self._logfile))
        self.stream.write(f'Running {suite.countTestCases()} tests.\n')
        startedPool = await poolStarter.start(self._reactor)
        condition = partial(shouldContinue, untilFailure)

        @countingCalls
        async def runAndReport(n: int) -> DistReporter:
            if untilFailure:
                self.stream.write(f'Test Pass {n + 1}\n')
            result = self._makeResult()
            if self._exitFirst:
                casesCondition = lambda _: result.original.wasSuccessful()
            else:
                casesCondition = lambda _: True
            await runTests(startedPool, takeWhile(casesCondition, testCases), result, self._driveWorker)
            self.writeResults(result)
            return result
        try:
            return await iterateWhile(condition, runAndReport)
        finally:
            await startedPool.join()

    def _run(self, test: Union[TestCase, TestSuite], untilFailure: bool) -> IReporter:
        if False:
            return 10
        result: Union[Failure, DistReporter, None] = None
        reactorStopping: bool = False
        testsInProgress: Deferred[object]

        def capture(r: Union[Failure, DistReporter]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal result
            result = r

        def maybeStopTests() -> Optional[Deferred[object]]:
            if False:
                while True:
                    i = 10
            nonlocal reactorStopping
            reactorStopping = True
            if result is None:
                testsInProgress.cancel()
                return testsInProgress
            return None

        def maybeStopReactor(result: object) -> object:
            if False:
                while True:
                    i = 10
            if not reactorStopping:
                self._reactor.stop()
            return result
        self._reactor.addSystemEventTrigger('before', 'shutdown', maybeStopTests)
        testsInProgress = Deferred.fromCoroutine(self.runAsync(test, untilFailure)).addBoth(capture).addBoth(maybeStopReactor)
        self._reactor.run()
        if isinstance(result, Failure):
            result.raiseException()
        assert isinstance(result, DistReporter), f'{result} is not DistReporter'
        return cast(IReporter, result.original)

    def run(self, test: Union[TestCase, TestSuite]) -> IReporter:
        if False:
            for i in range(10):
                print('nop')
        '\n        Run a reactor and a test suite.\n\n        @param test: The test or suite to run.\n        '
        return self._run(test, untilFailure=False)

    def runUntilFailure(self, test: Union[TestCase, TestSuite]) -> IReporter:
        if False:
            print('Hello World!')
        '\n        Run the tests with local worker processes until they fail.\n\n        @param test: The test or suite to run.\n        '
        return self._run(test, untilFailure=True)
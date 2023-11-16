"""
Tests for L{twisted.python.threadpool}
"""
import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest

class Synchronization:
    failures = 0

    def __init__(self, N, waiting):
        if False:
            while True:
                i = 10
        self.N = N
        self.waiting = waiting
        self.lock = threading.Lock()
        self.runs = []

    def run(self):
        if False:
            print('Hello World!')
        if self.lock.acquire(False):
            if not len(self.runs) % 5:
                time.sleep(0.0002)
            self.lock.release()
        else:
            self.failures += 1
        self.lock.acquire()
        self.runs.append(None)
        if len(self.runs) == self.N:
            self.waiting.release()
        self.lock.release()
    synchronized = ['run']
threadable.synchronize(Synchronization)

class ThreadPoolTests(unittest.SynchronousTestCase):
    """
    Test threadpools.
    """

    def getTimeout(self):
        if False:
            print('Hello World!')
        '\n        Return number of seconds to wait before giving up.\n        '
        return 5

    def _waitForLock(self, lock):
        if False:
            print('Hello World!')
        items = range(1000000)
        for i in items:
            if lock.acquire(False):
                break
            time.sleep(1e-05)
        else:
            self.fail('A long time passed without succeeding')

    def test_attributes(self):
        if False:
            while True:
                i = 10
        '\n        L{ThreadPool.min} and L{ThreadPool.max} are set to the values passed to\n        L{ThreadPool.__init__}.\n        '
        pool = threadpool.ThreadPool(12, 22)
        self.assertEqual(pool.min, 12)
        self.assertEqual(pool.max, 22)

    def test_start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ThreadPool.start} creates the minimum number of threads specified.\n        '
        pool = threadpool.ThreadPool(0, 5)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(len(pool.threads), 0)
        pool = threadpool.ThreadPool(3, 10)
        self.assertEqual(len(pool.threads), 0)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(len(pool.threads), 3)

    def test_adjustingWhenPoolStopped(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ThreadPool.adjustPoolsize} only modifies the pool size and does not\n        start new workers while the pool is not running.\n        '
        pool = threadpool.ThreadPool(0, 5)
        pool.start()
        pool.stop()
        pool.adjustPoolsize(2)
        self.assertEqual(len(pool.threads), 0)

    def test_threadCreationArguments(self):
        if False:
            return 10
        "\n        Test that creating threads in the threadpool with application-level\n        objects as arguments doesn't results in those objects never being\n        freed, with the thread maintaining a reference to them as long as it\n        exists.\n        "
        tp = threadpool.ThreadPool(0, 1)
        tp.start()
        self.addCleanup(tp.stop)
        self.assertEqual(tp.threads, [])

        def worker(arg):
            if False:
                i = 10
                return i + 15
            pass

        class Dumb:
            pass
        unique = Dumb()
        workerRef = weakref.ref(worker)
        uniqueRef = weakref.ref(unique)
        tp.callInThread(worker, unique)
        event = threading.Event()
        tp.callInThread(event.set)
        event.wait(self.getTimeout())
        del worker
        del unique
        gc.collect()
        self.assertIsNone(uniqueRef())
        self.assertIsNone(workerRef())

    def test_threadCreationArgumentsCallInThreadWithCallback(self):
        if False:
            while True:
                i = 10
        '\n        As C{test_threadCreationArguments} above, but for\n        callInThreadWithCallback.\n        '
        tp = threadpool.ThreadPool(0, 1)
        tp.start()
        self.addCleanup(tp.stop)
        self.assertEqual(tp.threads, [])
        refdict = {}
        onResultWait = threading.Event()
        onResultDone = threading.Event()
        resultRef = []

        def onResult(success, result):
            if False:
                return 10
            gc.collect()
            onResultWait.wait(self.getTimeout())
            refdict['workerRef'] = workerRef()
            refdict['uniqueRef'] = uniqueRef()
            onResultDone.set()
            resultRef.append(weakref.ref(result))

        def worker(arg, test):
            if False:
                i = 10
                return i + 15
            return Dumb()

        class Dumb:
            pass
        unique = Dumb()
        onResultRef = weakref.ref(onResult)
        workerRef = weakref.ref(worker)
        uniqueRef = weakref.ref(unique)
        tp.callInThreadWithCallback(onResult, worker, unique, test=unique)
        del worker
        del unique
        onResultWait.set()
        onResultDone.wait(self.getTimeout())
        gc.collect()
        self.assertIsNone(uniqueRef())
        self.assertIsNone(workerRef())
        del onResult
        gc.collect()
        self.assertIsNone(onResultRef())
        self.assertIsNone(resultRef[0]())
        self.assertEqual(list(refdict.values()), [None, None])

    def test_persistence(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Threadpools can be pickled and unpickled, which should preserve the\n        number of threads and other parameters.\n        '
        pool = threadpool.ThreadPool(7, 20)
        self.assertEqual(pool.min, 7)
        self.assertEqual(pool.max, 20)
        copy = pickle.loads(pickle.dumps(pool))
        self.assertEqual(copy.min, 7)
        self.assertEqual(copy.max, 20)

    def _threadpoolTest(self, method):
        if False:
            print('Hello World!')
        '\n        Test synchronization of calls made with C{method}, which should be\n        one of the mechanisms of the threadpool to execute work in threads.\n        '
        N = 10
        tp = threadpool.ThreadPool()
        tp.start()
        self.addCleanup(tp.stop)
        waiting = threading.Lock()
        waiting.acquire()
        actor = Synchronization(N, waiting)
        for i in range(N):
            method(tp, actor)
        self._waitForLock(waiting)
        self.assertFalse(actor.failures, f'run() re-entered {actor.failures} times')

    def test_callInThread(self):
        if False:
            i = 10
            return i + 15
        '\n        Call C{_threadpoolTest} with C{callInThread}.\n        '
        return self._threadpoolTest(lambda tp, actor: tp.callInThread(actor.run))

    def test_callInThreadException(self):
        if False:
            return 10
        '\n        L{ThreadPool.callInThread} logs exceptions raised by the callable it\n        is passed.\n        '

        class NewError(Exception):
            pass

        def raiseError():
            if False:
                while True:
                    i = 10
            raise NewError()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThread(raiseError)
        tp.start()
        tp.stop()
        errors = self.flushLoggedErrors(NewError)
        self.assertEqual(len(errors), 1)

    def test_callInThreadWithCallback(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a\n        two-tuple of C{(True, result)} where C{result} is the value returned\n        by the callable supplied.\n        '
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            if False:
                print('Hello World!')
            waiter.release()
            results.append(success)
            results.append(result)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, lambda : 'test')
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        self.assertTrue(results[0])
        self.assertEqual(results[1], 'test')

    def test_callInThreadWithCallbackExceptionInCallback(self):
        if False:
            return 10
        '\n        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a\n        two-tuple of C{(False, failure)} where C{failure} represents the\n        exception raised by the callable supplied.\n        '

        class NewError(Exception):
            pass

        def raiseError():
            if False:
                return 10
            raise NewError()
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            if False:
                for i in range(10):
                    print('nop')
            waiter.release()
            results.append(success)
            results.append(result)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, raiseError)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        self.assertFalse(results[0])
        self.assertIsInstance(results[1], failure.Failure)
        self.assertTrue(issubclass(results[1].type, NewError))

    def test_callInThreadWithCallbackExceptionInOnResult(self):
        if False:
            return 10
        '\n        L{ThreadPool.callInThreadWithCallback} logs the exception raised by\n        C{onResult}.\n        '

        class NewError(Exception):
            pass
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            if False:
                for i in range(10):
                    print('nop')
            results.append(success)
            results.append(result)
            raise NewError()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, lambda : None)
        tp.callInThread(waiter.release)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        errors = self.flushLoggedErrors(NewError)
        self.assertEqual(len(errors), 1)
        self.assertTrue(results[0])
        self.assertIsNone(results[1])

    def test_callbackThread(self):
        if False:
            print('Hello World!')
        '\n        L{ThreadPool.callInThreadWithCallback} calls the function it is\n        given and the C{onResult} callback in the same thread.\n        '
        threadIds = []
        event = threading.Event()

        def onResult(success, result):
            if False:
                for i in range(10):
                    print('nop')
            threadIds.append(threading.current_thread().ident)
            event.set()

        def func():
            if False:
                while True:
                    i = 10
            threadIds.append(threading.current_thread().ident)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, func)
        tp.start()
        self.addCleanup(tp.stop)
        event.wait(self.getTimeout())
        self.assertEqual(len(threadIds), 2)
        self.assertEqual(threadIds[0], threadIds[1])

    def test_callbackContext(self):
        if False:
            while True:
                i = 10
        '\n        The context L{ThreadPool.callInThreadWithCallback} is invoked in is\n        shared by the context the callable and C{onResult} callback are\n        invoked in.\n        '
        myctx = context.theContextTracker.currentContext().contexts[-1]
        myctx['testing'] = 'this must be present'
        contexts = []
        event = threading.Event()

        def onResult(success, result):
            if False:
                print('Hello World!')
            ctx = context.theContextTracker.currentContext().contexts[-1]
            contexts.append(ctx)
            event.set()

        def func():
            if False:
                print('Hello World!')
            ctx = context.theContextTracker.currentContext().contexts[-1]
            contexts.append(ctx)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, func)
        tp.start()
        self.addCleanup(tp.stop)
        event.wait(self.getTimeout())
        self.assertEqual(len(contexts), 2)
        self.assertEqual(myctx, contexts[0])
        self.assertEqual(myctx, contexts[1])

    def test_existingWork(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Work added to the threadpool before its start should be executed once\n        the threadpool is started: this is ensured by trying to release a lock\n        previously acquired.\n        '
        waiter = threading.Lock()
        waiter.acquire()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThread(waiter.release)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()

    def test_workerStateTransition(self):
        if False:
            while True:
                i = 10
        '\n        As the worker receives and completes work, it transitions between\n        the working and waiting states.\n        '
        pool = threadpool.ThreadPool(0, 1)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(pool.workers, 0)
        self.assertEqual(len(pool.waiters), 0)
        self.assertEqual(len(pool.working), 0)
        threadWorking = threading.Event()
        threadFinish = threading.Event()

        def _thread():
            if False:
                for i in range(10):
                    print('nop')
            threadWorking.set()
            threadFinish.wait(10)
        pool.callInThread(_thread)
        threadWorking.wait(10)
        self.assertEqual(pool.workers, 1)
        self.assertEqual(len(pool.waiters), 0)
        self.assertEqual(len(pool.working), 1)
        threadFinish.set()
        while not len(pool.waiters):
            time.sleep(0.0005)
        self.assertEqual(len(pool.waiters), 1)
        self.assertEqual(len(pool.working), 0)

    def test_q(self) -> None:
        if False:
            print('Hello World!')
        "\n        There is a property '_queue' for legacy purposes\n        "
        pool = threadpool.ThreadPool(0, 1)
        self.assertEqual(pool._queue.qsize(), 0)

class RaceConditionTests(unittest.SynchronousTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.threadpool = threadpool.ThreadPool(0, 10)
        self.event = threading.Event()
        self.threadpool.start()

        def done():
            if False:
                for i in range(10):
                    print('nop')
            self.threadpool.stop()
            del self.threadpool
        self.addCleanup(done)

    def getTimeout(self):
        if False:
            return 10
        '\n        A reasonable number of seconds to time out.\n        '
        return 5

    def test_synchronization(self):
        if False:
            print('Hello World!')
        '\n        If multiple threads are waiting on an event (via blocking on something\n        in a callable passed to L{threadpool.ThreadPool.callInThread}), and\n        there is spare capacity in the threadpool, sending another callable\n        which will cause those to un-block to\n        L{threadpool.ThreadPool.callInThread} will reliably run that callable\n        and un-block the blocked threads promptly.\n\n        @note: This is not really a unit test, it is a stress-test.  You may\n            need to run it with C{trial -u} to fail reliably if there is a\n            problem.  It is very hard to regression-test for this particular\n            bug - one where the thread pool may consider itself as having\n            "enough capacity" when it really needs to spin up a new thread if\n            it possibly can - in a deterministic way, since the bug can only be\n            provoked by subtle race conditions.\n        '
        timeout = self.getTimeout()
        self.threadpool.callInThread(self.event.set)
        self.event.wait(timeout)
        self.event.clear()
        for i in range(3):
            self.threadpool.callInThread(self.event.wait)
        self.threadpool.callInThread(self.event.set)
        self.event.wait(timeout)
        if not self.event.isSet():
            self.event.set()
            self.fail("'set' did not run in thread; timed out waiting on 'wait'.")

class MemoryPool(threadpool.ThreadPool):
    """
    A deterministic threadpool that uses in-memory data structures to queue
    work rather than threads to execute work.
    """

    def __init__(self, coordinator, failTest, newWorker, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize this L{MemoryPool} with a test case.\n\n        @param coordinator: a worker used to coordinate work in the L{Team}\n            underlying this threadpool.\n        @type coordinator: L{twisted._threads.IExclusiveWorker}\n\n        @param failTest: A 1-argument callable taking an exception and raising\n            a test-failure exception.\n        @type failTest: 1-argument callable taking (L{Failure}) and raising\n            L{unittest.FailTest}.\n\n        @param newWorker: a 0-argument callable that produces a new\n            L{twisted._threads.IWorker} provider on each invocation.\n        @type newWorker: 0-argument callable returning\n            L{twisted._threads.IWorker}.\n        '
        self._coordinator = coordinator
        self._failTest = failTest
        self._newWorker = newWorker
        threadpool.ThreadPool.__init__(self, *args, **kwargs)

    def _pool(self, currentLimit, threadFactory):
        if False:
            while True:
                i = 10
        '\n        Override testing hook to create a deterministic threadpool.\n\n        @param currentLimit: A 1-argument callable which returns the current\n            threadpool size limit.\n\n        @param threadFactory: ignored in this invocation; a 0-argument callable\n            that would produce a thread.\n\n        @return: a L{Team} backed by the coordinator and worker passed to\n            L{MemoryPool.__init__}.\n        '

        def respectLimit():
            if False:
                for i in range(10):
                    print('nop')
            stats = team.statistics()
            if stats.busyWorkerCount + stats.idleWorkerCount >= currentLimit():
                return None
            return self._newWorker()
        team = Team(coordinator=self._coordinator, createWorker=respectLimit, logException=self._failTest)
        return team

class PoolHelper:
    """
    A L{PoolHelper} constructs a L{threadpool.ThreadPool} that doesn't actually
    use threads, by using the internal interfaces in L{twisted._threads}.

    @ivar performCoordination: a 0-argument callable that will perform one unit
        of "coordination" - work involved in delegating work to other threads -
        and return L{True} if it did any work, L{False} otherwise.

    @ivar workers: the workers which represent the threads within the pool -
        the workers other than the coordinator.
    @type workers: L{list} of 2-tuple of (L{IWorker}, C{workPerformer}) where
        C{workPerformer} is a 0-argument callable like C{performCoordination}.

    @ivar threadpool: a modified L{threadpool.ThreadPool} to test.
    @type threadpool: L{MemoryPool}
    """

    def __init__(self, testCase, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Create a L{PoolHelper}.\n\n        @param testCase: a test case attached to this helper.\n\n        @type args: The arguments passed to a L{threadpool.ThreadPool}.\n\n        @type kwargs: The arguments passed to a L{threadpool.ThreadPool}\n        '
        (coordinator, self.performCoordination) = createMemoryWorker()
        self.workers = []

        def newWorker():
            if False:
                return 10
            self.workers.append(createMemoryWorker())
            return self.workers[-1][0]
        self.threadpool = MemoryPool(coordinator, testCase.fail, newWorker, *args, **kwargs)

    def performAllCoordination(self):
        if False:
            print('Hello World!')
        '\n        Perform all currently scheduled "coordination", which is the work\n        involved in delegating work to other threads.\n        '
        while self.performCoordination():
            pass

class MemoryBackedTests(unittest.SynchronousTestCase):
    """
    Tests using L{PoolHelper} to deterministically test properties of the
    threadpool implementation.
    """

    def test_workBeforeStarting(self):
        if False:
            i = 10
            return i + 15
        "\n        If a threadpool is told to do work before starting, then upon starting\n        up, it will start enough workers to handle all of the enqueued work\n        that it's been given.\n        "
        helper = PoolHelper(self, 0, 10)
        n = 5
        for x in range(n):
            helper.threadpool.callInThread(lambda : None)
        helper.performAllCoordination()
        self.assertEqual(helper.workers, [])
        helper.threadpool.start()
        helper.performAllCoordination()
        self.assertEqual(len(helper.workers), n)

    def test_tooMuchWorkBeforeStarting(self):
        if False:
            while True:
                i = 10
        '\n        If the amount of work before starting exceeds the maximum number of\n        threads allowed to the threadpool, only the maximum count will be\n        started.\n        '
        helper = PoolHelper(self, 0, 10)
        n = 50
        for x in range(n):
            helper.threadpool.callInThread(lambda : None)
        helper.performAllCoordination()
        self.assertEqual(helper.workers, [])
        helper.threadpool.start()
        helper.performAllCoordination()
        self.assertEqual(len(helper.workers), helper.threadpool.max)
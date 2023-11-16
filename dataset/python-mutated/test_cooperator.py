"""
This module contains tests for L{twisted.internet.task.Cooperator} and
related functionality.
"""
from twisted.internet import defer, reactor, task
from twisted.trial import unittest

class FakeDelayedCall:
    """
    Fake delayed call which lets us simulate the scheduler.
    """

    def __init__(self, func):
        if False:
            return 10
        '\n        A function to run, later.\n        '
        self.func = func
        self.cancelled = False

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Don't run my function later.\n        "
        self.cancelled = True

class FakeScheduler:
    """
    A fake scheduler for testing against.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a fake scheduler with a list of work to do.\n        '
        self.work = []

    def __call__(self, thunk):
        if False:
            for i in range(10):
                print('nop')
        '\n        Schedule a unit of work to be done later.\n        '
        unit = FakeDelayedCall(thunk)
        self.work.append(unit)
        return unit

    def pump(self):
        if False:
            print('Hello World!')
        '\n        Do all of the work that is currently available to be done.\n        '
        (work, self.work) = (self.work, [])
        for unit in work:
            if not unit.cancelled:
                unit.func()

class CooperatorTests(unittest.TestCase):
    RESULT = 'done'

    def ebIter(self, err):
        if False:
            return 10
        err.trap(task.SchedulerStopped)
        return self.RESULT

    def cbIter(self, ign):
        if False:
            i = 10
            return i + 15
        self.fail()

    def testStoppedRejectsNewTasks(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that Cooperators refuse new tasks when they have been stopped.\n        '

        def testwith(stuff):
            if False:
                while True:
                    i = 10
            c = task.Cooperator()
            c.stop()
            d = c.coiterate(iter(()), stuff)
            d.addCallback(self.cbIter)
            d.addErrback(self.ebIter)
            return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))
        return testwith(None).addCallback(lambda ign: testwith(defer.Deferred()))

    def testStopRunning(self):
        if False:
            print('Hello World!')
        '\n        Test that a running iterator will not run to completion when the\n        cooperator is stopped.\n        '
        c = task.Cooperator()

        def myiter():
            if False:
                return 10
            yield from range(3)
        myiter.value = -1
        d = c.coiterate(myiter())
        d.addCallback(self.cbIter)
        d.addErrback(self.ebIter)
        c.stop()

        def doasserts(result):
            if False:
                i = 10
                return i + 15
            self.assertEqual(result, self.RESULT)
            self.assertEqual(myiter.value, -1)
        d.addCallback(doasserts)
        return d

    def testStopOutstanding(self):
        if False:
            while True:
                i = 10
        '\n        An iterator run with L{Cooperator.coiterate} paused on a L{Deferred}\n        yielded by that iterator will fire its own L{Deferred} (the one\n        returned by C{coiterate}) when L{Cooperator.stop} is called.\n        '
        testControlD = defer.Deferred()
        outstandingD = defer.Deferred()

        def myiter():
            if False:
                return 10
            reactor.callLater(0, testControlD.callback, None)
            yield outstandingD
            self.fail()
        c = task.Cooperator()
        d = c.coiterate(myiter())

        def stopAndGo(ign):
            if False:
                print('Hello World!')
            c.stop()
            outstandingD.callback('arglebargle')
        testControlD.addCallback(stopAndGo)
        d.addCallback(self.cbIter)
        d.addErrback(self.ebIter)
        return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))

    def testUnexpectedError(self):
        if False:
            while True:
                i = 10
        c = task.Cooperator()

        def myiter():
            if False:
                while True:
                    i = 10
            if False:
                yield None
            else:
                raise RuntimeError()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testUnexpectedErrorActuallyLater(self):
        if False:
            return 10

        def myiter():
            if False:
                print('Hello World!')
            D = defer.Deferred()
            reactor.callLater(0, D.errback, RuntimeError())
            yield D
        c = task.Cooperator()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testUnexpectedErrorNotActuallyLater(self):
        if False:
            for i in range(10):
                print('nop')

        def myiter():
            if False:
                while True:
                    i = 10
            yield defer.fail(RuntimeError())
        c = task.Cooperator()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testCooperation(self):
        if False:
            print('Hello World!')
        L = []

        def myiter(things):
            if False:
                i = 10
                return i + 15
            for th in things:
                L.append(th)
                yield None
        groupsOfThings = ['abc', (1, 2, 3), 'def', (4, 5, 6)]
        c = task.Cooperator()
        tasks = []
        for stuff in groupsOfThings:
            tasks.append(c.coiterate(myiter(stuff)))
        return defer.DeferredList(tasks).addCallback(lambda ign: self.assertEqual(tuple(L), sum(zip(*groupsOfThings), ())))

    def testResourceExhaustion(self):
        if False:
            for i in range(10):
                print('nop')
        output = []

        def myiter():
            if False:
                while True:
                    i = 10
            for i in range(100):
                output.append(i)
                if i == 9:
                    _TPF.stopped = True
                yield i

        class _TPF:
            stopped = False

            def __call__(self):
                if False:
                    i = 10
                    return i + 15
                return self.stopped
        c = task.Cooperator(terminationPredicateFactory=_TPF)
        c.coiterate(myiter()).addErrback(self.ebIter)
        c._delayedCall.cancel()
        c._tick()
        c.stop()
        self.assertTrue(_TPF.stopped)
        self.assertEqual(output, list(range(10)))

    def testCallbackReCoiterate(self):
        if False:
            return 10
        '\n        If a callback to a deferred returned by coiterate calls coiterate on\n        the same Cooperator, we should make sure to only do the minimal amount\n        of scheduling work.  (This test was added to demonstrate a specific bug\n        that was found while writing the scheduler.)\n        '
        calls = []

        class FakeCall:

            def __init__(self, func):
                if False:
                    print('Hello World!')
                self.func = func

            def __repr__(self) -> str:
                if False:
                    for i in range(10):
                        print('nop')
                return f'<FakeCall {self.func!r}>'

        def sched(f):
            if False:
                while True:
                    i = 10
            self.assertFalse(calls, repr(calls))
            calls.append(FakeCall(f))
            return calls[-1]
        c = task.Cooperator(scheduler=sched, terminationPredicateFactory=lambda : lambda : True)
        d = c.coiterate(iter(()))
        done = []

        def anotherTask(ign):
            if False:
                i = 10
                return i + 15
            c.coiterate(iter(())).addBoth(done.append)
        d.addCallback(anotherTask)
        work = 0
        while not done:
            work += 1
            while calls:
                calls.pop(0).func()
                work += 1
            if work > 50:
                self.fail('Cooperator took too long')

    def test_removingLastTaskStopsScheduledCall(self):
        if False:
            while True:
                i = 10
        "\n        If the last task in a Cooperator is removed, the scheduled call for\n        the next tick is cancelled, since it is no longer necessary.\n\n        This behavior is useful for tests that want to assert they have left\n        no reactor state behind when they're done.\n        "
        calls = [None]

        def sched(f):
            if False:
                for i in range(10):
                    print('nop')
            calls[0] = FakeDelayedCall(f)
            return calls[0]
        coop = task.Cooperator(scheduler=sched)
        task1 = coop.cooperate(iter([1, 2]))
        task2 = coop.cooperate(iter([1, 2]))
        self.assertEqual(calls[0].func, coop._tick)
        task1.stop()
        self.assertFalse(calls[0].cancelled)
        self.assertEqual(coop._delayedCall, calls[0])
        task2.stop()
        self.assertTrue(calls[0].cancelled)
        self.assertIsNone(coop._delayedCall)
        coop.cooperate(iter([1, 2]))
        self.assertFalse(calls[0].cancelled)
        self.assertEqual(coop._delayedCall, calls[0])

    def test_runningWhenStarted(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Cooperator.running} reports C{True} if the L{Cooperator}\n        was started on creation.\n        '
        c = task.Cooperator()
        self.assertTrue(c.running)

    def test_runningWhenNotStarted(self):
        if False:
            print('Hello World!')
        '\n        L{Cooperator.running} reports C{False} if the L{Cooperator}\n        has not been started.\n        '
        c = task.Cooperator(started=False)
        self.assertFalse(c.running)

    def test_runningWhenRunning(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Cooperator.running} reports C{True} when the L{Cooperator}\n        is running.\n        '
        c = task.Cooperator(started=False)
        c.start()
        self.addCleanup(c.stop)
        self.assertTrue(c.running)

    def test_runningWhenStopped(self):
        if False:
            return 10
        '\n        L{Cooperator.running} reports C{False} after the L{Cooperator}\n        has been stopped.\n        '
        c = task.Cooperator(started=False)
        c.start()
        c.stop()
        self.assertFalse(c.running)

class UnhandledException(Exception):
    """
    An exception that should go unhandled.
    """

class AliasTests(unittest.TestCase):
    """
    Integration test to verify that the global singleton aliases do what
    they're supposed to.
    """

    def test_cooperate(self):
        if False:
            return 10
        '\n        L{twisted.internet.task.cooperate} ought to run the generator that it is\n        '
        d = defer.Deferred()

        def doit():
            if False:
                print('Hello World!')
            yield 1
            yield 2
            yield 3
            d.callback('yay')
        it = doit()
        theTask = task.cooperate(it)
        self.assertIn(theTask, task._theCooperator._tasks)
        return d

class RunStateTests(unittest.TestCase):
    """
    Tests to verify the behavior of L{CooperativeTask.pause},
    L{CooperativeTask.resume}, L{CooperativeTask.stop}, exhausting the
    underlying iterator, and their interactions with each other.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a cooperator with a fake scheduler and a termination predicate\n        that ensures only one unit of work will take place per tick.\n        '
        self._doDeferNext = False
        self._doStopNext = False
        self._doDieNext = False
        self.work = []
        self.scheduler = FakeScheduler()
        self.cooperator = task.Cooperator(scheduler=self.scheduler, terminationPredicateFactory=lambda : lambda : True)
        self.task = self.cooperator.cooperate(self.worker())
        self.cooperator.start()

    def worker(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is a sample generator which yields Deferreds when we are testing\n        deferral and an ascending integer count otherwise.\n        '
        i = 0
        while True:
            i += 1
            if self._doDeferNext:
                self._doDeferNext = False
                d = defer.Deferred()
                self.work.append(d)
                yield d
            elif self._doStopNext:
                return
            elif self._doDieNext:
                raise UnhandledException()
            else:
                self.work.append(i)
                yield i

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Drop references to interesting parts of the fixture to allow Deferred\n        errors to be noticed when things start failing.\n        '
        del self.task
        del self.scheduler

    def deferNext(self):
        if False:
            print('Hello World!')
        '\n        Defer the next result from my worker iterator.\n        '
        self._doDeferNext = True

    def stopNext(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make the next result from my worker iterator be completion (raising\n        StopIteration).\n        '
        self._doStopNext = True

    def dieNext(self):
        if False:
            print('Hello World!')
        '\n        Make the next result from my worker iterator be raising an\n        L{UnhandledException}.\n        '

        def ignoreUnhandled(failure):
            if False:
                return 10
            failure.trap(UnhandledException)
            return None
        self._doDieNext = True

    def test_pauseResume(self):
        if False:
            print('Hello World!')
        "\n        Cooperators should stop running their tasks when they're paused, and\n        start again when they're resumed.\n        "
        self.scheduler.pump()
        self.assertEqual(self.work, [1])
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2])
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2])
        self.task.resume()
        self.assertEqual(self.work, [1, 2])
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2, 3])

    def test_resumeNotPaused(self):
        if False:
            i = 10
            return i + 15
        '\n        L{CooperativeTask.resume} should raise a L{TaskNotPaused} exception if\n        it was not paused; e.g. if L{CooperativeTask.pause} was not invoked\n        more times than L{CooperativeTask.resume} on that object.\n        '
        self.assertRaises(task.NotPaused, self.task.resume)
        self.task.pause()
        self.task.resume()
        self.assertRaises(task.NotPaused, self.task.resume)

    def test_pauseTwice(self):
        if False:
            i = 10
            return i + 15
        '\n        Pauses on tasks should behave like a stack. If a task is paused twice,\n        it needs to be resumed twice.\n        '
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [1])

    def test_pauseWhileDeferred(self):
        if False:
            i = 10
            return i + 15
        '\n        C{pause()}ing a task while it is waiting on an outstanding\n        L{defer.Deferred} should put the task into a state where the\n        outstanding L{defer.Deferred} must be called back I{and} the task is\n        C{resume}d before it will continue processing.\n        '
        self.deferNext()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.assertIsInstance(self.work[0], defer.Deferred)
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.work[0].callback('STUFF!')
        self.scheduler.pump()
        self.assertEqual(len(self.work), 2)
        self.assertEqual(self.work[1], 2)

    def test_whenDone(self):
        if False:
            print('Hello World!')
        "\n        L{CooperativeTask.whenDone} returns a Deferred which fires when the\n        Cooperator's iterator is exhausted.  It returns a new Deferred each\n        time it is called; callbacks added to other invocations will not modify\n        the value that subsequent invocations will fire with.\n        "
        deferred1 = self.task.whenDone()
        deferred2 = self.task.whenDone()
        results1 = []
        results2 = []
        final1 = []
        final2 = []

        def callbackOne(result):
            if False:
                i = 10
                return i + 15
            results1.append(result)
            return 1

        def callbackTwo(result):
            if False:
                return 10
            results2.append(result)
            return 2
        deferred1.addCallback(callbackOne)
        deferred2.addCallback(callbackTwo)
        deferred1.addCallback(final1.append)
        deferred2.addCallback(final2.append)
        self.stopNext()
        self.scheduler.pump()
        self.assertEqual(len(results1), 1)
        self.assertEqual(len(results2), 1)
        self.assertIs(results1[0], self.task._iterator)
        self.assertIs(results2[0], self.task._iterator)
        self.assertEqual(final1, [1])
        self.assertEqual(final2, [2])

    def test_whenDoneError(self):
        if False:
            return 10
        "\n        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that will fail\n        when the iterable's C{next} method raises an exception, with that\n        exception.\n        "
        deferred1 = self.task.whenDone()
        results = []
        deferred1.addErrback(results.append)
        self.dieNext()
        self.scheduler.pump()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check(UnhandledException), UnhandledException)

    def test_whenDoneStop(self):
        if False:
            return 10
        '\n        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that fails with\n        L{TaskStopped} when the C{stop} method is called on that\n        L{CooperativeTask}.\n        '
        deferred1 = self.task.whenDone()
        errors = []
        deferred1.addErrback(errors.append)
        self.task.stop()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].check(task.TaskStopped), task.TaskStopped)

    def test_whenDoneAlreadyDone(self):
        if False:
            print('Hello World!')
        '\n        L{CooperativeTask.whenDone} will return a L{defer.Deferred} that will\n        succeed immediately if its iterator has already completed.\n        '
        self.stopNext()
        self.scheduler.pump()
        results = []
        self.task.whenDone().addCallback(results.append)
        self.assertEqual(results, [self.task._iterator])

    def test_stopStops(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        C{stop()}ping a task should cause it to be removed from the run just as\n        C{pause()}ing, with the distinction that C{resume()} will raise a\n        L{TaskStopped} exception.\n        '
        self.task.stop()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 0)
        self.assertRaises(task.TaskStopped, self.task.stop)
        self.assertRaises(task.TaskStopped, self.task.pause)
        self.scheduler.pump()
        self.assertEqual(self.work, [])

    def test_pauseStopResume(self):
        if False:
            while True:
                i = 10
        "\n        C{resume()}ing a paused, stopped task should be a no-op; it should not\n        raise an exception, because it's paused, but neither should it actually\n        do more work from the task.\n        "
        self.task.pause()
        self.task.stop()
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [])

    def test_stopDeferred(self):
        if False:
            return 10
        '\n        As a corrolary of the interaction of C{pause()} and C{unpause()},\n        C{stop()}ping a task which is waiting on a L{Deferred} should cause the\n        task to gracefully shut down, meaning that it should not be unpaused\n        when the deferred fires.\n        '
        self.deferNext()
        self.scheduler.pump()
        d = self.work.pop()
        self.assertEqual(self.task._pauseCount, 1)
        results = []
        d.addBoth(results.append)
        self.scheduler.pump()
        self.task.stop()
        self.scheduler.pump()
        d.callback(7)
        self.scheduler.pump()
        self.assertEqual(results, [None])
        self.assertEqual(self.work, [])

    def test_stopExhausted(self):
        if False:
            i = 10
            return i + 15
        '\n        C{stop()}ping a L{CooperativeTask} whose iterator has been exhausted\n        should raise L{TaskDone}.\n        '
        self.stopNext()
        self.scheduler.pump()
        self.assertRaises(task.TaskDone, self.task.stop)

    def test_stopErrored(self):
        if False:
            return 10
        '\n        C{stop()}ping a L{CooperativeTask} whose iterator has encountered an\n        error should raise L{TaskFailed}.\n        '
        self.dieNext()
        self.scheduler.pump()
        self.assertRaises(task.TaskFailed, self.task.stop)

    def test_stopCooperatorReentrancy(self):
        if False:
            i = 10
            return i + 15
        "\n        If a callback of a L{Deferred} from L{CooperativeTask.whenDone} calls\n        C{Cooperator.stop} on its L{CooperativeTask._cooperator}, the\n        L{Cooperator} will stop, but the L{CooperativeTask} whose callback is\n        calling C{stop} should already be considered 'stopped' by the time the\n        callback is running, and therefore removed from the\n        L{CoooperativeTask}.\n        "
        callbackPhases = []

        def stopit(result):
            if False:
                print('Hello World!')
            callbackPhases.append(result)
            self.cooperator.stop()
            callbackPhases.append('done')
        self.task.whenDone().addCallback(stopit)
        self.stopNext()
        self.scheduler.pump()
        self.assertEqual(callbackPhases, [self.task._iterator, 'done'])
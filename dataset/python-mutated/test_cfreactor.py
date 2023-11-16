from typing import TYPE_CHECKING, List
from twisted.trial.unittest import SynchronousTestCase
from .reactormixins import ReactorBuilder
if TYPE_CHECKING:
    fakeBase = SynchronousTestCase
else:
    fakeBase = object

def noop() -> None:
    if False:
        return 10
    '\n    Do-nothing callable. Stub for testing.\n    '
noop()

class CoreFoundationSpecificTests(ReactorBuilder, fakeBase):
    """
    Tests for platform interactions of the CoreFoundation-based reactor.
    """
    _reactors = ['twisted.internet.cfreactor.CFReactor']

    def test_whiteboxStopSimulating(self) -> None:
        if False:
            while True:
                i = 10
        "\n        CFReactor's simulation timer is None after CFReactor crashes.\n        "
        r = self.buildReactor()
        r.callLater(0, r.crash)
        r.callLater(100, noop)
        self.runReactor(r)
        self.assertIs(r._currentSimulator, None)

    def test_callLaterLeakage(self) -> None:
        if False:
            print('Hello World!')
        '\n        callLater should not leak global state into CoreFoundation which will\n        be invoked by a different reactor running the main loop.\n\n        @note: this test may actually be usable for other reactors as well, so\n            we may wish to promote it to ensure this invariant across other\n            foreign-main-loop reactors.\n        '
        r = self.buildReactor()
        delayed = r.callLater(0, noop)
        r2 = self.buildReactor()

        def stopBlocking() -> None:
            if False:
                return 10
            r2.callLater(0, r2stop)

        def r2stop() -> None:
            if False:
                print('Hello World!')
            r2.stop()
        r2.callLater(0, stopBlocking)
        self.runReactor(r2)
        self.assertEqual(r.getDelayedCalls(), [delayed])

    def test_whiteboxIterate(self) -> None:
        if False:
            while True:
                i = 10
        "\n        C{.iterate()} should remove the CFTimer that will run Twisted's\n        callLaters from the loop, even if one is still pending.  We test this\n        state indirectly with a white-box assertion by verifying the\n        C{_currentSimulator} is set to C{None}, since CoreFoundation does not\n        allow us to enumerate all active timers or sources.\n        "
        r = self.buildReactor()
        x: List[int] = []
        r.callLater(0, x.append, 1)
        delayed = r.callLater(100, noop)
        r.iterate()
        self.assertIs(r._currentSimulator, None)
        self.assertEqual(r.getDelayedCalls(), [delayed])
        self.assertEqual(x, [1])

    def test_noTimers(self) -> None:
        if False:
            print('Hello World!')
        '\n        The loop can wake up just fine even if there are no timers in it.\n        '
        r = self.buildReactor()
        stopped = []

        def doStop() -> None:
            if False:
                while True:
                    i = 10
            r.stop()
            stopped.append('yes')

        def sleepThenStop() -> None:
            if False:
                print('Hello World!')
            r.callFromThread(doStop)
        r.callLater(0, r.callInThread, sleepThenStop)
        r.run()
        self.assertEqual(stopped, ['yes'])
globals().update(CoreFoundationSpecificTests.makeTestCaseClasses())
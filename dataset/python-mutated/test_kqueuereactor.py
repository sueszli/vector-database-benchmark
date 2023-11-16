"""
Tests for L{twisted.internet.kqueuereactor}.
"""
from __future__ import annotations
import errno
from zope.interface import implementer
from twisted.trial.unittest import TestCase
try:
    from twisted.internet.kqreactor import KQueueReactor, _IKQueue
    kqueueSkip = None
except ImportError:
    kqueueSkip = 'KQueue not available.'

def _fakeKEvent(*args: object, **kwargs: object) -> None:
    if False:
        print('Hello World!')
    '\n    Do nothing.\n    '

def makeFakeKQueue(testKQueue: object, testKEvent: object) -> _IKQueue:
    if False:
        print('Hello World!')
    '\n    Create a fake that implements L{_IKQueue}.\n\n    @param testKQueue: Something that acts like L{select.kqueue}.\n    @param testKEvent: Something that acts like L{select.kevent}.\n    @return: An implementation of L{_IKQueue} that includes C{testKQueue} and\n        C{testKEvent}.\n    '

    @implementer(_IKQueue)
    class FakeKQueue:
        kqueue = testKQueue
        kevent = testKEvent
    return FakeKQueue()

class KQueueTests(TestCase):
    """
    These are tests for L{KQueueReactor}'s implementation, not its real world
    behaviour. For that, look at
    L{twisted.internet.test.reactormixins.ReactorBuilder}.
    """
    skip = kqueueSkip

    def test_EINTR(self) -> None:
        if False:
            return 10
        '\n        L{KQueueReactor} handles L{errno.EINTR} in C{doKEvent} by returning.\n        '

        class FakeKQueue:
            """
            A fake KQueue that raises L{errno.EINTR} when C{control} is called,
            like a real KQueue would if it was interrupted.
            """

            def control(self, *args: object, **kwargs: object) -> None:
                if False:
                    return 10
                raise OSError(errno.EINTR, 'Interrupted')
        reactor = KQueueReactor(makeFakeKQueue(FakeKQueue, _fakeKEvent))
        reactor.doKEvent(0)
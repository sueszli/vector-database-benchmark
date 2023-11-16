"""
Tests for L{twisted.internet.epollreactor}.
"""
from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
try:
    from twisted.internet import epollreactor
except ImportError:
    epollreactor = None

class Descriptor:
    """
    Records reads and writes, as if it were a C{FileDescriptor}.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.events = []

    def fileno(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def doRead(self):
        if False:
            return 10
        self.events.append('read')

    def doWrite(self):
        if False:
            print('Hello World!')
        self.events.append('write')

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        reason.trap(ConnectionDone)
        self.events.append('lost')

@skipIf(not epollreactor, 'epoll not supported in this environment.')
class ContinuousPollingTests(TestCase):
    """
    L{_ContinuousPolling} can be used to read and write from C{FileDescriptor}
    objects.
    """

    def test_addReader(self):
        if False:
            while True:
                i = 10
        '\n        Adding a reader when there was previously no reader starts up a\n        C{LoopingCall}.\n        '
        poller = _ContinuousPolling(Clock())
        self.assertIsNone(poller._loop)
        reader = object()
        self.assertFalse(poller.isReading(reader))
        poller.addReader(reader)
        self.assertIsNotNone(poller._loop)
        self.assertTrue(poller._loop.running)
        self.assertIs(poller._loop.clock, poller._reactor)
        self.assertTrue(poller.isReading(reader))

    def test_addWriter(self):
        if False:
            return 10
        '\n        Adding a writer when there was previously no writer starts up a\n        C{LoopingCall}.\n        '
        poller = _ContinuousPolling(Clock())
        self.assertIsNone(poller._loop)
        writer = object()
        self.assertFalse(poller.isWriting(writer))
        poller.addWriter(writer)
        self.assertIsNotNone(poller._loop)
        self.assertTrue(poller._loop.running)
        self.assertIs(poller._loop.clock, poller._reactor)
        self.assertTrue(poller.isWriting(writer))

    def test_removeReader(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removing a reader stops the C{LoopingCall}.\n        '
        poller = _ContinuousPolling(Clock())
        reader = object()
        poller.addReader(reader)
        poller.removeReader(reader)
        self.assertIsNone(poller._loop)
        self.assertEqual(poller._reactor.getDelayedCalls(), [])
        self.assertFalse(poller.isReading(reader))

    def test_removeWriter(self):
        if False:
            while True:
                i = 10
        '\n        Removing a writer stops the C{LoopingCall}.\n        '
        poller = _ContinuousPolling(Clock())
        writer = object()
        poller.addWriter(writer)
        poller.removeWriter(writer)
        self.assertIsNone(poller._loop)
        self.assertEqual(poller._reactor.getDelayedCalls(), [])
        self.assertFalse(poller.isWriting(writer))

    def test_removeUnknown(self):
        if False:
            print('Hello World!')
        '\n        Removing unknown readers and writers silently does nothing.\n        '
        poller = _ContinuousPolling(Clock())
        poller.removeWriter(object())
        poller.removeReader(object())

    def test_multipleReadersAndWriters(self):
        if False:
            print('Hello World!')
        '\n        Adding multiple readers and writers results in a single\n        C{LoopingCall}.\n        '
        poller = _ContinuousPolling(Clock())
        writer = object()
        poller.addWriter(writer)
        self.assertIsNotNone(poller._loop)
        poller.addWriter(object())
        self.assertIsNotNone(poller._loop)
        poller.addReader(object())
        self.assertIsNotNone(poller._loop)
        poller.addReader(object())
        poller.removeWriter(writer)
        self.assertIsNotNone(poller._loop)
        self.assertTrue(poller._loop.running)
        self.assertEqual(len(poller._reactor.getDelayedCalls()), 1)

    def test_readerPolling(self):
        if False:
            print('Hello World!')
        '\n        Adding a reader causes its C{doRead} to be called every 1\n        milliseconds.\n        '
        reactor = Clock()
        poller = _ContinuousPolling(reactor)
        desc = Descriptor()
        poller.addReader(desc)
        self.assertEqual(desc.events, [])
        reactor.advance(1e-05)
        self.assertEqual(desc.events, ['read'])
        reactor.advance(1e-05)
        self.assertEqual(desc.events, ['read', 'read'])
        reactor.advance(1e-05)
        self.assertEqual(desc.events, ['read', 'read', 'read'])

    def test_writerPolling(self):
        if False:
            while True:
                i = 10
        '\n        Adding a writer causes its C{doWrite} to be called every 1\n        milliseconds.\n        '
        reactor = Clock()
        poller = _ContinuousPolling(reactor)
        desc = Descriptor()
        poller.addWriter(desc)
        self.assertEqual(desc.events, [])
        reactor.advance(0.001)
        self.assertEqual(desc.events, ['write'])
        reactor.advance(0.001)
        self.assertEqual(desc.events, ['write', 'write'])
        reactor.advance(0.001)
        self.assertEqual(desc.events, ['write', 'write', 'write'])

    def test_connectionLostOnRead(self):
        if False:
            while True:
                i = 10
        '\n        If a C{doRead} returns a value indicating disconnection,\n        C{connectionLost} is called on it.\n        '
        reactor = Clock()
        poller = _ContinuousPolling(reactor)
        desc = Descriptor()
        desc.doRead = lambda : ConnectionDone()
        poller.addReader(desc)
        self.assertEqual(desc.events, [])
        reactor.advance(0.001)
        self.assertEqual(desc.events, ['lost'])

    def test_connectionLostOnWrite(self):
        if False:
            print('Hello World!')
        '\n        If a C{doWrite} returns a value indicating disconnection,\n        C{connectionLost} is called on it.\n        '
        reactor = Clock()
        poller = _ContinuousPolling(reactor)
        desc = Descriptor()
        desc.doWrite = lambda : ConnectionDone()
        poller.addWriter(desc)
        self.assertEqual(desc.events, [])
        reactor.advance(0.001)
        self.assertEqual(desc.events, ['lost'])

    def test_removeAll(self):
        if False:
            return 10
        '\n        L{_ContinuousPolling.removeAll} removes all descriptors and returns\n        the readers and writers.\n        '
        poller = _ContinuousPolling(Clock())
        reader = object()
        writer = object()
        both = object()
        poller.addReader(reader)
        poller.addReader(both)
        poller.addWriter(writer)
        poller.addWriter(both)
        removed = poller.removeAll()
        self.assertEqual(poller.getReaders(), [])
        self.assertEqual(poller.getWriters(), [])
        self.assertEqual(len(removed), 3)
        self.assertEqual(set(removed), {reader, writer, both})

    def test_getReaders(self):
        if False:
            i = 10
            return i + 15
        '\n        L{_ContinuousPolling.getReaders} returns a list of the read\n        descriptors.\n        '
        poller = _ContinuousPolling(Clock())
        reader = object()
        poller.addReader(reader)
        self.assertIn(reader, poller.getReaders())

    def test_getWriters(self):
        if False:
            print('Hello World!')
        '\n        L{_ContinuousPolling.getWriters} returns a list of the write\n        descriptors.\n        '
        poller = _ContinuousPolling(Clock())
        writer = object()
        poller.addWriter(writer)
        self.assertIn(writer, poller.getWriters())
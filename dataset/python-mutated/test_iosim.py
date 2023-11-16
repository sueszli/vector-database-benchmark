"""
Tests for L{twisted.test.iosim}.
"""
from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase

class FakeTransportTests(TestCase):
    """
    Tests for L{FakeTransport}.
    """

    def test_connectionSerial(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Each L{FakeTransport} receives a serial number that uniquely identifies\n        it.\n        '
        a = FakeTransport(object(), True)
        b = FakeTransport(object(), False)
        self.assertIsInstance(a.serial, int)
        self.assertIsInstance(b.serial, int)
        self.assertNotEqual(a.serial, b.serial)

    def test_writeSequence(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{FakeTransport.writeSequence} will write a sequence of L{bytes} to the\n        transport.\n        '
        a = FakeTransport(object(), False)
        a.write(b'a')
        a.writeSequence([b'b', b'c', b'd'])
        self.assertEqual(b''.join(a.stream), b'abcd')

    def test_writeAfterClose(self) -> None:
        if False:
            return 10
        '\n        L{FakeTransport.write} will accept writes after transport was closed,\n        but the data will be silently discarded.\n        '
        a = FakeTransport(object(), False)
        a.write(b'before')
        a.loseConnection()
        a.write(b'after')
        self.assertEqual(b''.join(a.stream), b'before')

@implementer(IPushProducer)
class StrictPushProducer:
    """
    An L{IPushProducer} implementation which produces nothing but enforces
    preconditions on its state transition methods.
    """
    _state = 'running'

    def stopProducing(self) -> None:
        if False:
            while True:
                i = 10
        if self._state == 'stopped':
            raise ValueError('Cannot stop already-stopped IPushProducer')
        self._state = 'stopped'

    def pauseProducing(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._state != 'running':
            raise ValueError(f'Cannot pause {self._state} IPushProducer')
        self._state = 'paused'

    def resumeProducing(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._state != 'paused':
            raise ValueError(f'Cannot resume {self._state} IPushProducer')
        self._state = 'running'

class StrictPushProducerTests(TestCase):
    """
    Tests for L{StrictPushProducer}.
    """

    def _initial(self) -> StrictPushProducer:
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: A new L{StrictPushProducer} which has not been through any state\n            changes.\n        '
        return StrictPushProducer()

    def _stopped(self) -> StrictPushProducer:
        if False:
            i = 10
            return i + 15
        '\n        @return: A new, stopped L{StrictPushProducer}.\n        '
        producer = StrictPushProducer()
        producer.stopProducing()
        return producer

    def _paused(self) -> StrictPushProducer:
        if False:
            while True:
                i = 10
        '\n        @return: A new, paused L{StrictPushProducer}.\n        '
        producer = StrictPushProducer()
        producer.pauseProducing()
        return producer

    def _resumed(self) -> StrictPushProducer:
        if False:
            i = 10
            return i + 15
        '\n        @return: A new L{StrictPushProducer} which has been paused and resumed.\n        '
        producer = StrictPushProducer()
        producer.pauseProducing()
        producer.resumeProducing()
        return producer

    def assertStopped(self, producer: StrictPushProducer) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that the given producer is in the stopped state.\n\n        @param producer: The producer to verify.\n        @type producer: L{StrictPushProducer}\n        '
        self.assertEqual(producer._state, 'stopped')

    def assertPaused(self, producer: StrictPushProducer) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that the given producer is in the paused state.\n\n        @param producer: The producer to verify.\n        @type producer: L{StrictPushProducer}\n        '
        self.assertEqual(producer._state, 'paused')

    def assertRunning(self, producer: StrictPushProducer) -> None:
        if False:
            return 10
        '\n        Assert that the given producer is in the running state.\n\n        @param producer: The producer to verify.\n        @type producer: L{StrictPushProducer}\n        '
        self.assertEqual(producer._state, 'running')

    def test_stopThenStop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{StrictPushProducer.stopProducing} raises L{ValueError} if called when\n        the producer is stopped.\n        '
        self.assertRaises(ValueError, self._stopped().stopProducing)

    def test_stopThenPause(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{StrictPushProducer.pauseProducing} raises L{ValueError} if called when\n        the producer is stopped.\n        '
        self.assertRaises(ValueError, self._stopped().pauseProducing)

    def test_stopThenResume(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{StrictPushProducer.resumeProducing} raises L{ValueError} if called when\n        the producer is stopped.\n        '
        self.assertRaises(ValueError, self._stopped().resumeProducing)

    def test_pauseThenStop(self) -> None:
        if False:
            return 10
        '\n        L{StrictPushProducer} is stopped if C{stopProducing} is called on a paused\n        producer.\n        '
        producer = self._paused()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_pauseThenPause(self) -> None:
        if False:
            return 10
        '\n        L{StrictPushProducer.pauseProducing} raises L{ValueError} if called on a\n        paused producer.\n        '
        producer = self._paused()
        self.assertRaises(ValueError, producer.pauseProducing)

    def test_pauseThenResume(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{StrictPushProducer} is resumed if C{resumeProducing} is called on a\n        paused producer.\n        '
        producer = self._paused()
        producer.resumeProducing()
        self.assertRunning(producer)

    def test_resumeThenStop(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{StrictPushProducer} is stopped if C{stopProducing} is called on a\n        resumed producer.\n        '
        producer = self._resumed()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_resumeThenPause(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{StrictPushProducer} is paused if C{pauseProducing} is called on a\n        resumed producer.\n        '
        producer = self._resumed()
        producer.pauseProducing()
        self.assertPaused(producer)

    def test_resumeThenResume(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{StrictPushProducer.resumeProducing} raises L{ValueError} if called on a\n        resumed producer.\n        '
        producer = self._resumed()
        self.assertRaises(ValueError, producer.resumeProducing)

    def test_stop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{StrictPushProducer} is stopped if C{stopProducing} is called in the\n        initial state.\n        '
        producer = self._initial()
        producer.stopProducing()
        self.assertStopped(producer)

    def test_pause(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{StrictPushProducer} is paused if C{pauseProducing} is called in the\n        initial state.\n        '
        producer = self._initial()
        producer.pauseProducing()
        self.assertPaused(producer)

    def test_resume(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{StrictPushProducer} raises L{ValueError} if C{resumeProducing} is called\n        in the initial state.\n        '
        producer = self._initial()
        self.assertRaises(ValueError, producer.resumeProducing)

class IOPumpTests(TestCase):
    """
    Tests for L{IOPump}.
    """

    def _testStreamingProducer(self, mode: Literal['server', 'client']) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Connect a couple protocol/transport pairs to an L{IOPump} and then pump\n        it.  Verify that a streaming producer registered with one of the\n        transports does not receive invalid L{IPushProducer} method calls and\n        ends in the right state.\n\n        @param mode: C{u"server"} to test a producer registered with the\n            server transport.  C{u"client"} to test a producer registered with\n            the client transport.\n        '
        serverProto = Protocol()
        serverTransport = FakeTransport(serverProto, isServer=True)
        clientProto = Protocol()
        clientTransport = FakeTransport(clientProto, isServer=False)
        pump = connect(serverProto, serverTransport, clientProto, clientTransport, greet=False)
        producer = StrictPushProducer()
        victim = {'server': serverTransport, 'client': clientTransport}[mode]
        victim.registerProducer(producer, streaming=True)
        pump.pump()
        self.assertEqual('running', producer._state)

    def test_serverStreamingProducer(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{IOPump.pump} does not call C{resumeProducing} on a L{IPushProducer}\n        (stream producer) registered with the server transport.\n        '
        self._testStreamingProducer(mode='server')

    def test_clientStreamingProducer(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{IOPump.pump} does not call C{resumeProducing} on a L{IPushProducer}\n        (stream producer) registered with the client transport.\n        '
        self._testStreamingProducer(mode='client')

    def test_timeAdvances(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{IOPump.pump} advances time in the given L{Clock}.\n        '
        time_passed = []
        clock = Clock()
        (_, _, pump) = connectedServerAndClient(Protocol, Protocol, clock=clock)
        clock.callLater(0, lambda : time_passed.append(True))
        self.assertFalse(time_passed)
        pump.pump()
        self.assertTrue(time_passed)
"""
Tests for L{twisted.internet._newtls}.
"""
from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import ConnectableProtocol, runProtocolsWithReactor
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import ContextGeneratingMixin, SSLCreator, StartTLSClientCreator, TLSMixin
from twisted.trial import unittest
try:
    from twisted.internet import _newtls as __newtls
    from twisted.protocols import tls
except ImportError:
    _newtls = None
else:
    _newtls = __newtls
from zope.interface import implementer

class BypassTLSTests(unittest.TestCase):
    """
    Tests for the L{_newtls._BypassTLS} class.
    """
    if not _newtls:
        skip = "Couldn't import _newtls, perhaps pyOpenSSL is old or missing"

    def test_loseConnectionPassThrough(self):
        if False:
            print('Hello World!')
        "\n        C{_BypassTLS.loseConnection} calls C{loseConnection} on the base\n        class, while preserving any default argument in the base class'\n        C{loseConnection} implementation.\n        "
        default = object()
        result = []

        class FakeTransport:

            def loseConnection(self, _connDone=default):
                if False:
                    print('Hello World!')
                result.append(_connDone)
        bypass = _newtls._BypassTLS(FakeTransport, FakeTransport())
        bypass.loseConnection()
        self.assertEqual(result, [default])
        notDefault = object()
        bypass.loseConnection(notDefault)
        self.assertEqual(result, [default, notDefault])

class FakeProducer:
    """
    A producer that does nothing.
    """

    def pauseProducing(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def resumeProducing(self):
        if False:
            while True:
                i = 10
        pass

    def stopProducing(self):
        if False:
            i = 10
            return i + 15
        pass

@implementer(interfaces.IHandshakeListener)
class ProducerProtocol(ConnectableProtocol):
    """
    Register a producer, unregister it, and verify the producer hooks up to
    innards of C{TLSMemoryBIOProtocol}.
    """

    def __init__(self, producer, result):
        if False:
            return 10
        self.producer = producer
        self.result = result

    def handshakeCompleted(self):
        if False:
            i = 10
            return i + 15
        if not isinstance(self.transport.protocol, tls.BufferingTLSTransport):
            raise RuntimeError('TLSMemoryBIOProtocol not hooked up.')
        self.transport.registerProducer(self.producer, True)
        self.result.append(self.transport.protocol._producer._producer)
        self.transport.unregisterProducer()
        self.result.append(self.transport.protocol._producer)
        self.transport.loseConnection()

class ProducerTestsMixin(ReactorBuilder, TLSMixin, ContextGeneratingMixin):
    """
    Test the new TLS code integrates C{TLSMemoryBIOProtocol} correctly.
    """
    if not _newtls:
        skip = 'Could not import twisted.internet._newtls'

    def test_producerSSLFromStart(self):
        if False:
            print('Hello World!')
        '\n        C{registerProducer} and C{unregisterProducer} on TLS transports\n        created as SSL from the get go are passed to the\n        C{TLSMemoryBIOProtocol}, not the underlying transport directly.\n        '
        result = []
        producer = FakeProducer()
        runProtocolsWithReactor(self, ConnectableProtocol(), ProducerProtocol(producer, result), SSLCreator())
        self.assertEqual(result, [producer, None])

    def test_producerAfterStartTLS(self):
        if False:
            while True:
                i = 10
        '\n        C{registerProducer} and C{unregisterProducer} on TLS transports\n        created by C{startTLS} are passed to the C{TLSMemoryBIOProtocol}, not\n        the underlying transport directly.\n        '
        result = []
        producer = FakeProducer()
        runProtocolsWithReactor(self, ConnectableProtocol(), ProducerProtocol(producer, result), StartTLSClientCreator())
        self.assertEqual(result, [producer, None])

    def startTLSAfterRegisterProducer(self, streaming):
        if False:
            print('Hello World!')
        '\n        When a producer is registered, and then startTLS is called,\n        the producer is re-registered with the C{TLSMemoryBIOProtocol}.\n        '
        clientContext = self.getClientContext()
        serverContext = self.getServerContext()
        result = []
        producer = FakeProducer()

        class RegisterTLSProtocol(ConnectableProtocol):

            def connectionMade(self):
                if False:
                    print('Hello World!')
                self.transport.registerProducer(producer, streaming)
                self.transport.startTLS(serverContext)
                if streaming:
                    result.append(self.transport.protocol._producer._producer)
                    result.append(self.transport.producer._producer)
                else:
                    result.append(self.transport.protocol._producer._producer._producer)
                    result.append(self.transport.producer._producer._producer)
                self.transport.unregisterProducer()
                self.transport.loseConnection()

        class StartTLSProtocol(ConnectableProtocol):

            def connectionMade(self):
                if False:
                    return 10
                self.transport.startTLS(clientContext)
        runProtocolsWithReactor(self, RegisterTLSProtocol(), StartTLSProtocol(), TCPCreator())
        self.assertEqual(result, [producer, producer])

    def test_startTLSAfterRegisterProducerStreaming(self):
        if False:
            print('Hello World!')
        '\n        When a streaming producer is registered, and then startTLS is called,\n        the producer is re-registered with the C{TLSMemoryBIOProtocol}.\n        '
        self.startTLSAfterRegisterProducer(True)

    def test_startTLSAfterRegisterProducerNonStreaming(self):
        if False:
            return 10
        '\n        When a non-streaming producer is registered, and then startTLS is\n        called, the producer is re-registered with the\n        C{TLSMemoryBIOProtocol}.\n        '
        self.startTLSAfterRegisterProducer(False)
globals().update(ProducerTestsMixin.makeTestCaseClasses())
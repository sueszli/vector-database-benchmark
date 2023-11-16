from typing import Tuple
from twisted.internet.protocol import Protocol
from twisted.test.proto_helpers import AccumulatingProtocol, MemoryReactorClock
from synapse.logging import RemoteHandler
from tests.logging import LoggerCleanupMixin
from tests.server import FakeTransport, get_clock
from tests.unittest import TestCase
from tests.utils import checked_cast

def connect_logging_client(reactor: MemoryReactorClock, client_id: int) -> Tuple[Protocol, AccumulatingProtocol]:
    if False:
        while True:
            i = 10
    factory = reactor.tcpClients.pop(client_id)[2]
    client = factory.buildProtocol(None)
    server = AccumulatingProtocol()
    server.makeConnection(FakeTransport(client, reactor))
    client.makeConnection(FakeTransport(server, reactor, autoflush=False))
    return (client, server)

class RemoteHandlerTestCase(LoggerCleanupMixin, TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        (self.reactor, _) = get_clock()

    def test_log_output(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The remote handler delivers logs over TCP.\n        '
        handler = RemoteHandler('127.0.0.1', 9000, _reactor=self.reactor)
        logger = self.get_logger(handler)
        logger.info('Hello there, %s!', 'wally')
        (client, server) = connect_logging_client(self.reactor, 0)
        client_transport = checked_cast(FakeTransport, client.transport)
        client_transport.flush()
        logs = server.data.decode('utf8').splitlines()
        self.assertEqual(len(logs), 1)
        self.assertEqual(server.data.count(b'\n'), 1)
        self.assertEqual(logs[0], 'Hello there, wally!')

    def test_log_backpressure_debug(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        When backpressure is hit, DEBUG logs will be shed.\n        '
        handler = RemoteHandler('127.0.0.1', 9000, maximum_buffer=10, _reactor=self.reactor)
        logger = self.get_logger(handler)
        for i in range(3):
            logger.debug('debug %s' % (i,))
        for i in range(7):
            logger.info('info %s' % (i,))
        logger.debug('too much debug')
        (client, server) = connect_logging_client(self.reactor, 0)
        client_transport = checked_cast(FakeTransport, client.transport)
        client_transport.flush()
        logs = server.data.splitlines()
        self.assertEqual(len(logs), 7)
        self.assertNotIn(b'debug', server.data)

    def test_log_backpressure_info(self) -> None:
        if False:
            return 10
        '\n        When backpressure is hit, DEBUG and INFO logs will be shed.\n        '
        handler = RemoteHandler('127.0.0.1', 9000, maximum_buffer=10, _reactor=self.reactor)
        logger = self.get_logger(handler)
        for i in range(3):
            logger.debug('debug %s' % (i,))
        for i in range(10):
            logger.warning('warn %s' % (i,))
        for i in range(3):
            logger.info('info %s' % (i,))
        logger.debug('too much debug')
        (client, server) = connect_logging_client(self.reactor, 0)
        client_transport = checked_cast(FakeTransport, client.transport)
        client_transport.flush()
        logs = server.data.splitlines()
        self.assertEqual(len(logs), 10)
        self.assertNotIn(b'debug', server.data)
        self.assertNotIn(b'info', server.data)

    def test_log_backpressure_cut_middle(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        When backpressure is hit, and no more DEBUG and INFOs cannot be culled,\n        it will cut the middle messages out.\n        '
        handler = RemoteHandler('127.0.0.1', 9000, maximum_buffer=10, _reactor=self.reactor)
        logger = self.get_logger(handler)
        for i in range(20):
            logger.warning('warn %s' % (i,))
        (client, server) = connect_logging_client(self.reactor, 0)
        client_transport = checked_cast(FakeTransport, client.transport)
        client_transport.flush()
        logs = server.data.decode('utf8').splitlines()
        self.assertEqual(['warn %s' % (i,) for i in range(5)] + ['warn %s' % (i,) for i in range(15, 20)], logs)

    def test_cancel_connection(self) -> None:
        if False:
            return 10
        '\n        Gracefully handle the connection being cancelled.\n        '
        handler = RemoteHandler('127.0.0.1', 9000, maximum_buffer=10, _reactor=self.reactor)
        logger = self.get_logger(handler)
        logger.info('Hello there, %s!', 'wally')
        handler.close()
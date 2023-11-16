import logging
import sys
import traceback
from collections import deque
from ipaddress import IPv4Address, IPv6Address, ip_address
from math import floor
from typing import Callable, Deque, Optional
import attr
from zope.interface import implementer
from twisted.application.internet import ClientService
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint, TCP6ClientEndpoint
from twisted.internet.interfaces import IPushProducer, IReactorTCP, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.tcp import Connection
from twisted.python.failure import Failure
logger = logging.getLogger(__name__)

@attr.s(slots=True, auto_attribs=True)
@implementer(IPushProducer)
class LogProducer:
    """
    An IPushProducer that writes logs from its buffer to its transport when it
    is resumed.

    Args:
        buffer: Log buffer to read logs from.
        transport: Transport to write to.
        format: A callable to format the log record to a string.
    """
    transport: Connection
    _format: Callable[[logging.LogRecord], str]
    _buffer: Deque[logging.LogRecord]
    _paused: bool = attr.ib(default=False, init=False)

    def pauseProducing(self) -> None:
        if False:
            while True:
                i = 10
        self._paused = True

    def stopProducing(self) -> None:
        if False:
            return 10
        self._paused = True
        self._buffer = deque()

    def resumeProducing(self) -> None:
        if False:
            return 10
        self._paused = False
        while self._paused is False and (self._buffer and self.transport.connected):
            try:
                record = self._buffer.popleft()
                msg = self._format(record)
                self.transport.write(msg.encode('utf8'))
                self.transport.write(b'\n')
            except Exception:
                traceback.print_exc(file=sys.__stderr__)
                break

class RemoteHandler(logging.Handler):
    """
    An logging handler that writes logs to a TCP target.

    Args:
        host: The host of the logging target.
        port: The logging target's port.
        maximum_buffer: The maximum buffer size.
    """

    def __init__(self, host: str, port: int, maximum_buffer: int=1000, level: int=logging.NOTSET, _reactor: Optional[IReactorTCP]=None):
        if False:
            while True:
                i = 10
        super().__init__(level=level)
        self.host = host
        self.port = port
        self.maximum_buffer = maximum_buffer
        self._buffer: Deque[logging.LogRecord] = deque()
        self._connection_waiter: Optional[Deferred] = None
        self._producer: Optional[LogProducer] = None
        if _reactor is None:
            from twisted.internet import reactor
            _reactor = reactor
        try:
            ip = ip_address(self.host)
            if isinstance(ip, IPv4Address):
                endpoint: IStreamClientEndpoint = TCP4ClientEndpoint(_reactor, self.host, self.port)
            elif isinstance(ip, IPv6Address):
                endpoint = TCP6ClientEndpoint(_reactor, self.host, self.port)
            else:
                raise ValueError('Unknown IP address provided: %s' % (self.host,))
        except ValueError:
            endpoint = HostnameEndpoint(_reactor, self.host, self.port)
        factory = Factory.forProtocol(Protocol)
        self._service = ClientService(endpoint, factory, clock=_reactor)
        self._service.startService()
        self._stopping = False
        self._connect()

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self._stopping = True
        self._service.stopService()

    def _connect(self) -> None:
        if False:
            return 10
        '\n        Triggers an attempt to connect then write to the remote if not already writing.\n        '
        if self._connection_waiter:
            return

        def fail(failure: Failure) -> None:
            if False:
                return 10
            if failure.check(CancelledError) and self._stopping:
                return
            failure.printTraceback(file=sys.__stderr__)
            self._connection_waiter = None
            self._connect()

        def writer(result: Protocol) -> None:
            if False:
                i = 10
                return i + 15
            transport: Connection = result.transport
            if self._producer and transport is self._producer.transport:
                self._producer.resumeProducing()
                self._connection_waiter = None
                return
            if self._producer:
                self._producer.stopProducing()
            self._producer = LogProducer(buffer=self._buffer, transport=transport, format=self.format)
            transport.registerProducer(self._producer, True)
            self._producer.resumeProducing()
            self._connection_waiter = None
        deferred: Deferred = self._service.whenConnected(failAfterFailures=1)
        deferred.addCallbacks(writer, fail)
        self._connection_waiter = deferred

    def _handle_pressure(self) -> None:
        if False:
            print('Hello World!')
        '\n        Handle backpressure by shedding records.\n\n        The buffer will, in this order, until the buffer is below the maximum:\n            - Shed DEBUG records.\n            - Shed INFO records.\n            - Shed the middle 50% of the records.\n        '
        if len(self._buffer) <= self.maximum_buffer:
            return
        self._buffer = deque(filter(lambda record: record.levelno > logging.DEBUG, self._buffer))
        if len(self._buffer) <= self.maximum_buffer:
            return
        self._buffer = deque(filter(lambda record: record.levelno > logging.INFO, self._buffer))
        if len(self._buffer) <= self.maximum_buffer:
            return
        buffer_split = floor(self.maximum_buffer / 2)
        old_buffer = self._buffer
        self._buffer = deque()
        for _ in range(buffer_split):
            self._buffer.append(old_buffer.popleft())
        end_buffer = []
        for _ in range(buffer_split):
            end_buffer.append(old_buffer.pop())
        self._buffer.extend(reversed(end_buffer))

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            print('Hello World!')
        self._buffer.append(record)
        try:
            self._handle_pressure()
        except Exception:
            self._buffer.clear()
            logger.warning('Failed clearing backpressure')
        self._connect()
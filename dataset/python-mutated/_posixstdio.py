"""Standard input/out/err support.

Future Plans::

    support for stderr, perhaps
    Rewrite to use the reactor instead of an ad-hoc mechanism for connecting
        protocols to transport.

Maintainer: James Y Knight
"""
from zope.interface import implementer
from twisted.internet import error, interfaces, process
from twisted.python import failure, log

@implementer(interfaces.IAddress)
class PipeAddress:
    pass

@implementer(interfaces.ITransport, interfaces.IProducer, interfaces.IConsumer, interfaces.IHalfCloseableDescriptor)
class StandardIO:
    _reader = None
    _writer = None
    disconnected = False
    disconnecting = False

    def __init__(self, proto, stdin=0, stdout=1, reactor=None):
        if False:
            while True:
                i = 10
        if reactor is None:
            from twisted.internet import reactor
        self.protocol = proto
        self._writer = process.ProcessWriter(reactor, self, 'write', stdout)
        self._reader = process.ProcessReader(reactor, self, 'read', stdin)
        self._reader.startReading()
        self.protocol.makeConnection(self)

    def loseWriteConnection(self):
        if False:
            print('Hello World!')
        if self._writer is not None:
            self._writer.loseConnection()

    def write(self, data):
        if False:
            i = 10
            return i + 15
        if self._writer is not None:
            self._writer.write(data)

    def writeSequence(self, data):
        if False:
            i = 10
            return i + 15
        if self._writer is not None:
            self._writer.writeSequence(data)

    def loseConnection(self):
        if False:
            i = 10
            return i + 15
        self.disconnecting = True
        if self._writer is not None:
            self._writer.loseConnection()
        if self._reader is not None:
            self._reader.stopReading()

    def getPeer(self):
        if False:
            i = 10
            return i + 15
        return PipeAddress()

    def getHost(self):
        if False:
            while True:
                i = 10
        return PipeAddress()

    def childDataReceived(self, fd, data):
        if False:
            i = 10
            return i + 15
        self.protocol.dataReceived(data)

    def childConnectionLost(self, fd, reason):
        if False:
            for i in range(10):
                print('nop')
        if self.disconnected:
            return
        if reason.value.__class__ == error.ConnectionDone:
            if fd == 'read':
                self._readConnectionLost(reason)
            else:
                self._writeConnectionLost(reason)
        else:
            self.connectionLost(reason)

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        self.disconnected = True
        _reader = self._reader
        _writer = self._writer
        protocol = self.protocol
        self._reader = self._writer = None
        self.protocol = None
        if _writer is not None and (not _writer.disconnected):
            _writer.connectionLost(reason)
        if _reader is not None and (not _reader.disconnected):
            _reader.connectionLost(reason)
        try:
            protocol.connectionLost(reason)
        except BaseException:
            log.err()

    def _writeConnectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        self._writer = None
        if self.disconnecting:
            self.connectionLost(reason)
            return
        p = interfaces.IHalfCloseableProtocol(self.protocol, None)
        if p:
            try:
                p.writeConnectionLost()
            except BaseException:
                log.err()
                self.connectionLost(failure.Failure())

    def _readConnectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        self._reader = None
        p = interfaces.IHalfCloseableProtocol(self.protocol, None)
        if p:
            try:
                p.readConnectionLost()
            except BaseException:
                log.err()
                self.connectionLost(failure.Failure())
        else:
            self.connectionLost(reason)

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        if self._writer is None:
            producer.stopProducing()
        else:
            self._writer.registerProducer(producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        if self._writer is not None:
            self._writer.unregisterProducer()

    def stopProducing(self):
        if False:
            while True:
                i = 10
        self.loseConnection()

    def pauseProducing(self):
        if False:
            return 10
        if self._reader is not None:
            self._reader.pauseProducing()

    def resumeProducing(self):
        if False:
            i = 10
            return i + 15
        if self._reader is not None:
            self._reader.resumeProducing()

    def stopReading(self):
        if False:
            for i in range(10):
                print('nop')
        "Compatibility only, don't use. Call pauseProducing."
        self.pauseProducing()

    def startReading(self):
        if False:
            for i in range(10):
                print('nop')
        "Compatibility only, don't use. Call resumeProducing."
        self.resumeProducing()

    def readConnectionLost(self, reason):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def writeConnectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()
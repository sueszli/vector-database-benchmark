"""
Windows-specific implementation of the L{twisted.internet.stdio} interface.
"""
import msvcrt
import os
from zope.interface import implementer
import win32api
from twisted.internet import _pollingfile, main
from twisted.internet.interfaces import IAddress, IConsumer, IHalfCloseableProtocol, IPushProducer, ITransport
from twisted.python.failure import Failure

@implementer(IAddress)
class Win32PipeAddress:
    pass

@implementer(ITransport, IConsumer, IPushProducer)
class StandardIO(_pollingfile._PollingTimer):
    disconnecting = False
    disconnected = False

    def __init__(self, proto, reactor=None):
        if False:
            i = 10
            return i + 15
        '\n        Start talking to standard IO with the given protocol.\n\n        Also, put it stdin/stdout/stderr into binary mode.\n        '
        if reactor is None:
            from twisted.internet import reactor
        for stdfd in range(0, 1, 2):
            msvcrt.setmode(stdfd, os.O_BINARY)
        _pollingfile._PollingTimer.__init__(self, reactor)
        self.proto = proto
        hstdin = win32api.GetStdHandle(win32api.STD_INPUT_HANDLE)
        hstdout = win32api.GetStdHandle(win32api.STD_OUTPUT_HANDLE)
        self.stdin = _pollingfile._PollableReadPipe(hstdin, self.dataReceived, self.readConnectionLost)
        self.stdout = _pollingfile._PollableWritePipe(hstdout, self.writeConnectionLost)
        self._addPollableResource(self.stdin)
        self._addPollableResource(self.stdout)
        self.proto.makeConnection(self)

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        self.proto.dataReceived(data)

    def readConnectionLost(self):
        if False:
            return 10
        if IHalfCloseableProtocol.providedBy(self.proto):
            self.proto.readConnectionLost()
        self.checkConnLost()

    def writeConnectionLost(self):
        if False:
            i = 10
            return i + 15
        if IHalfCloseableProtocol.providedBy(self.proto):
            self.proto.writeConnectionLost()
        self.checkConnLost()
    connsLost = 0

    def checkConnLost(self):
        if False:
            print('Hello World!')
        self.connsLost += 1
        if self.connsLost >= 2:
            self.disconnecting = True
            self.disconnected = True
            self.proto.connectionLost(Failure(main.CONNECTION_DONE))

    def write(self, data):
        if False:
            while True:
                i = 10
        self.stdout.write(data)

    def writeSequence(self, seq):
        if False:
            for i in range(10):
                print('nop')
        self.stdout.write(b''.join(seq))

    def loseConnection(self):
        if False:
            print('Hello World!')
        self.disconnecting = True
        self.stdin.close()
        self.stdout.close()

    def getPeer(self):
        if False:
            for i in range(10):
                print('nop')
        return Win32PipeAddress()

    def getHost(self):
        if False:
            while True:
                i = 10
        return Win32PipeAddress()

    def registerProducer(self, producer, streaming):
        if False:
            print('Hello World!')
        return self.stdout.registerProducer(producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        return self.stdout.unregisterProducer()

    def stopProducing(self):
        if False:
            while True:
                i = 10
        self.stdin.stopProducing()

    def pauseProducing(self):
        if False:
            for i in range(10):
                print('nop')
        self.stdin.pauseProducing()

    def resumeProducing(self):
        if False:
            return 10
        self.stdin.resumeProducing()
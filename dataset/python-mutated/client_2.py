from twisted.internet import defer, reactor
from twisted.protocols import basic

class ClientTimeoutError(Exception):
    pass

class RemoteCalculationClient(basic.LineReceiver):
    callLater = reactor.callLater
    timeOut = 60

    def __init__(self):
        if False:
            print('Hello World!')
        self.results = []

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        (d, callID) = self.results.pop(0)
        callID.cancel()
        d.callback(int(line))

    def _cancel(self, d):
        if False:
            print('Hello World!')
        d.errback(ClientTimeoutError())

    def _sendOperation(self, op, a, b):
        if False:
            for i in range(10):
                print('nop')
        d = defer.Deferred()
        callID = self.callLater(self.timeOut, self._cancel, d)
        self.results.append((d, callID))
        line = f'{op} {a} {b}'.encode()
        self.sendLine(line)
        return d

    def add(self, a, b):
        if False:
            print('Hello World!')
        return self._sendOperation('add', a, b)

    def subtract(self, a, b):
        if False:
            print('Hello World!')
        return self._sendOperation('subtract', a, b)

    def multiply(self, a, b):
        if False:
            print('Hello World!')
        return self._sendOperation('multiply', a, b)

    def divide(self, a, b):
        if False:
            return 10
        return self._sendOperation('divide', a, b)
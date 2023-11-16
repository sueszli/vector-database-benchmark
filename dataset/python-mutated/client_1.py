from twisted.internet import defer
from twisted.protocols import basic

class RemoteCalculationClient(basic.LineReceiver):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.results = []

    def lineReceived(self, line):
        if False:
            return 10
        d = self.results.pop(0)
        d.callback(int(line))

    def _sendOperation(self, op, a, b):
        if False:
            return 10
        d = defer.Deferred()
        self.results.append(d)
        line = f'{op} {a} {b}'.encode()
        self.sendLine(line)
        return d

    def add(self, a, b):
        if False:
            print('Hello World!')
        return self._sendOperation('add', a, b)

    def subtract(self, a, b):
        if False:
            i = 10
            return i + 15
        return self._sendOperation('subtract', a, b)

    def multiply(self, a, b):
        if False:
            return 10
        return self._sendOperation('multiply', a, b)

    def divide(self, a, b):
        if False:
            while True:
                i = 10
        return self._sendOperation('divide', a, b)
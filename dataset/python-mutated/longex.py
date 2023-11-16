"""Simple example of doing arbitrarily long calculations nicely in Twisted.

This is also a simple demonstration of twisted.protocols.basic.LineReceiver.
"""
from twisted.internet import reactor
from twisted.internet.protocol import ServerFactory
from twisted.protocols import basic

class LongMultiplicationProtocol(basic.LineReceiver):
    """A protocol for doing long multiplications.

    It receives a list of numbers (separated by whitespace) on a line, and
    writes back the answer.  The answer is calculated in chunks, so no one
    calculation should block for long enough to matter.
    """

    def connectionMade(self):
        if False:
            return 10
        self.workQueue = []

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        try:
            numbers = [int(num) for num in line.split()]
        except ValueError:
            self.sendLine(b'Error.')
            return
        if len(numbers) <= 1:
            self.sendLine(b'Error.')
            return
        self.workQueue.append(numbers)
        reactor.callLater(0, self.calcChunk)

    def calcChunk(self):
        if False:
            i = 10
            return i + 15
        if self.workQueue:
            work = self.workQueue[0]
            work[:2] = [work[0] * work[1]]
            if len(work) == 1:
                self.sendLine(str(work[0]).encode('ascii'))
                del self.workQueue[0]
            if self.workQueue:
                reactor.callLater(0, self.calcChunk)

class LongMultiplicationFactory(ServerFactory):
    protocol = LongMultiplicationProtocol
if __name__ == '__main__':
    import sys
    from twisted.python import log
    log.startLogging(sys.stdout)
    reactor.listenTCP(1234, LongMultiplicationFactory())
    reactor.run()
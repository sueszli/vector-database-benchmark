"""Throughput test."""
import sys
import time
from twisted.internet import protocol, reactor
from twisted.python import log
TIMES = 10000
S = '0123456789' * 1240
toReceive = len(S) * TIMES

class Sender(protocol.Protocol):

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        start()
        self.numSent = 0
        self.received = 0
        self.transport.registerProducer(self, 0)

    def stopProducing(self):
        if False:
            return 10
        pass

    def pauseProducing(self):
        if False:
            i = 10
            return i + 15
        pass

    def resumeProducing(self):
        if False:
            return 10
        self.numSent += 1
        self.transport.write(S)
        if self.numSent == TIMES:
            self.transport.unregisterProducer()
            self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        shutdown(self.numSent == TIMES)
started = None

def start():
    if False:
        return 10
    global started
    started = time.time()

def shutdown(success):
    if False:
        i = 10
        return i + 15
    if not success:
        raise SystemExit('failure or something')
    passed = time.time() - started
    print('Throughput (send): %s kbytes/sec' % (toReceive / passed / 1024))
    reactor.stop()

def main():
    if False:
        i = 10
        return i + 15
    f = protocol.ClientFactory()
    f.protocol = Sender
    reactor.connectTCP(sys.argv[1], int(sys.argv[2]), f)
    reactor.run()
if __name__ == '__main__':
    main()
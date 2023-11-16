from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.protocol import Protocol
from twisted.python import log
from twisted.web.client import Agent, ContentDecoderAgent, GzipDecoder

class BeginningPrinter(Protocol):

    def __init__(self, finished):
        if False:
            for i in range(10):
                print('nop')
        self.finished = finished
        self.remaining = 1024 * 10

    def dataReceived(self, bytes):
        if False:
            print('Hello World!')
        if self.remaining:
            display = bytes[:self.remaining]
            print('Some data received:')
            print(display)
            self.remaining -= len(display)

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        print('Finished receiving body:', reason.type, reason.value)
        self.finished.callback(None)

def printBody(response):
    if False:
        for i in range(10):
            print('nop')
    finished = Deferred()
    response.deliverBody(BeginningPrinter(finished))
    return finished

def main():
    if False:
        i = 10
        return i + 15
    agent = ContentDecoderAgent(Agent(reactor), [(b'gzip', GzipDecoder)])
    d = agent.request(b'GET', b'http://httpbin.org/gzip')
    d.addCallback(printBody)
    d.addErrback(log.err)
    d.addCallback(lambda ignored: reactor.stop())
    reactor.run()
if __name__ == '__main__':
    main()
from cStringIO import StringIO
from twisted.internet import protocol, reactor, utils
from twisted.python import failure

class FortuneQuoter(protocol.Protocol):
    fortune = '/usr/games/fortune'

    def connectionMade(self):
        if False:
            print('Hello World!')
        output = utils.getProcessOutput(self.fortune)
        output.addCallbacks(self.writeResponse, self.noResponse)

    def writeResponse(self, resp):
        if False:
            i = 10
            return i + 15
        self.transport.write(resp)
        self.transport.loseConnection()

    def noResponse(self, err):
        if False:
            while True:
                i = 10
        self.transport.loseConnection()
if __name__ == '__main__':
    f = protocol.Factory()
    f.protocol = FortuneQuoter
    reactor.listenTCP(10999, f)
    reactor.run()
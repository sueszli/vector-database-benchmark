from twisted.internet import endpoints, protocol, reactor
from twisted.protocols import basic

class FingerProtocol(basic.LineReceiver):

    def lineReceived(self, user):
        if False:
            while True:
                i = 10
        self.transport.write(self.factory.getUser(user) + b'\r\n')
        self.transport.loseConnection()

class FingerFactory(protocol.ServerFactory):
    protocol = FingerProtocol

    def getUser(self, user):
        if False:
            return 10
        return b'No such user'
fingerEndpoint = endpoints.serverFromString(reactor, 'tcp:1079')
fingerEndpoint.listen(FingerFactory())
reactor.run()
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

    def __init__(self, users):
        if False:
            for i in range(10):
                print('nop')
        self.users = users

    def getUser(self, user):
        if False:
            return 10
        return self.users.get(user, b'No such user')
fingerEndpoint = endpoints.serverFromString(reactor, 'tcp:1079')
fingerEndpoint.listen(FingerFactory({b'moshez': b'Happy and well'}))
reactor.run()
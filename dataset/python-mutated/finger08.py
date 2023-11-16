from twisted.internet import defer, endpoints, protocol, reactor
from twisted.protocols import basic

class FingerProtocol(basic.LineReceiver):

    def lineReceived(self, user):
        if False:
            i = 10
            return i + 15
        d = self.factory.getUser(user)

        def onError(err):
            if False:
                for i in range(10):
                    print('nop')
            return 'Internal error in server'
        d.addErrback(onError)

        def writeResponse(message):
            if False:
                i = 10
                return i + 15
            self.transport.write(message + b'\r\n')
            self.transport.loseConnection()
        d.addCallback(writeResponse)

class FingerFactory(protocol.ServerFactory):
    protocol = FingerProtocol

    def __init__(self, users):
        if False:
            i = 10
            return i + 15
        self.users = users

    def getUser(self, user):
        if False:
            while True:
                i = 10
        return defer.succeed(self.users.get(user, b'No such user'))
fingerEndpoint = endpoints.serverFromString(reactor, 'tcp:1079')
fingerEndpoint.listen(FingerFactory({b'moshez': b'Happy and well'}))
reactor.run()
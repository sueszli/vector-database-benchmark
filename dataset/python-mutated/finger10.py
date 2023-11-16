from twisted.internet import defer, endpoints, protocol, reactor, utils
from twisted.protocols import basic
from twisted.web import client

class FingerProtocol(basic.LineReceiver):

    def lineReceived(self, user):
        if False:
            print('Hello World!')
        d = self.factory.getUser(user)

        def onError(err):
            if False:
                for i in range(10):
                    print('nop')
            return b'Internal error in server'
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

    def __init__(self, prefix):
        if False:
            return 10
        self.prefix = prefix

    def getUser(self, user):
        if False:
            print('Hello World!')
        return client.getPage(self.prefix + user)
fingerEndpoint = endpoints.serverFromString(reactor, 'tcp:1079')
fingerEndpoint.listen(FingerFactory(prefix=b'http://livejournal.com/~'))
reactor.run()
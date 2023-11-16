from twisted.internet import defer, endpoints, protocol, reactor, utils
from twisted.protocols import basic

class FingerProtocol(basic.LineReceiver):

    def lineReceived(self, user):
        if False:
            for i in range(10):
                print('nop')
        d = self.factory.getUser(user)

        def onError(err):
            if False:
                while True:
                    i = 10
            return b'Internal error in server'
        d.addErrback(onError)

        def writeResponse(message):
            if False:
                print('Hello World!')
            self.transport.write(message + b'\r\n')
            self.transport.loseConnection()
        d.addCallback(writeResponse)

class FingerFactory(protocol.ServerFactory):
    protocol = FingerProtocol

    def getUser(self, user):
        if False:
            i = 10
            return i + 15
        return utils.getProcessOutput(b'finger', [user])
fingerEndpoint = endpoints.serverFromString(reactor, 'tcp:1079')
fingerEndpoint.listen(FingerFactory())
reactor.run()
from ampserver import Divide, Sum
from twisted.internet import defer, reactor
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.protocols.amp import AMP

def doMath():
    if False:
        print('Hello World!')
    destination = TCP4ClientEndpoint(reactor, '127.0.0.1', 1234)
    sumDeferred = connectProtocol(destination, AMP())

    def connected(ampProto):
        if False:
            while True:
                i = 10
        return ampProto.callRemote(Sum, a=13, b=81)
    sumDeferred.addCallback(connected)

    def summed(result):
        if False:
            print('Hello World!')
        return result['total']
    sumDeferred.addCallback(summed)
    divideDeferred = connectProtocol(destination, AMP())

    def connected(ampProto):
        if False:
            print('Hello World!')
        return ampProto.callRemote(Divide, numerator=1234, denominator=0)
    divideDeferred.addCallback(connected)

    def trapZero(result):
        if False:
            for i in range(10):
                print('nop')
        result.trap(ZeroDivisionError)
        print('Divided by zero: returning INF')
        return 1e309
    divideDeferred.addErrback(trapZero)

    def done(result):
        if False:
            return 10
        print('Done with math:', result)
        reactor.stop()
    defer.DeferredList([sumDeferred, divideDeferred]).addCallback(done)
if __name__ == '__main__':
    doMath()
    reactor.run()
from twisted.internet import endpoints, reactor
from twisted.spread import pb

def gotObject(object):
    if False:
        print('Hello World!')
    print('got object:', object)
    object.callRemote('getUser', 'moshez').addCallback(gotData)

def gotData(data):
    if False:
        for i in range(10):
            print('nop')
    print('server sent:', data)
    reactor.stop()

def gotNoObject(reason):
    if False:
        for i in range(10):
            print('nop')
    print('no object:', reason)
    reactor.stop()
factory = pb.PBClientFactory()
clientEndpoint = endpoints.clientFromString('tcp:127.0.0.1:8889')
clientEndpoint.connect(factory)
factory.getRootObject().addCallbacks(gotObject, gotNoObject)
reactor.run()
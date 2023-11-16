from twisted.internet import reactor
from twisted.spread import pb

def main():
    if False:
        return 10
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    d = factory.getRootObject()
    d.addCallbacks(got_obj)
    reactor.run()

def got_obj(obj):
    if False:
        print('Hello World!')
    d2 = obj.callRemote('broken')
    d2.addCallback(working)
    d2.addErrback(broken)

def working():
    if False:
        for i in range(10):
            print('nop')
    print("erm, it wasn't *supposed* to work..")

def broken(reason):
    if False:
        print('Hello World!')
    print('got remote Exception')
    print(' .__class__ =', reason.__class__)
    print(' .getErrorMessage() =', reason.getErrorMessage())
    print(' .type =', reason.type)
    reactor.stop()
main()
from twisted.internet import reactor
from twisted.spread import pb

class Two(pb.Referenceable):

    def remote_print(self, arg):
        if False:
            print('Hello World!')
        print('Two.print() called with', arg)

def main():
    if False:
        print('Hello World!')
    two = Two()
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    def1 = factory.getRootObject()
    def1.addCallback(got_obj, two)
    reactor.run()

def got_obj(obj, two):
    if False:
        i = 10
        return i + 15
    print('got One:', obj)
    print('giving it our two')
    obj.callRemote('takeTwo', two)
main()
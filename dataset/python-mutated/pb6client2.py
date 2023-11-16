from twisted.cred import credentials
from twisted.internet import reactor
from twisted.spread import pb

def main():
    if False:
        for i in range(10):
            print('nop')
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    def1 = factory.login(credentials.UsernamePassword('user2', 'pass2'))
    def1.addCallback(connected)
    reactor.run()

def connected(perspective):
    if False:
        print('Hello World!')
    print('got perspective2 ref:', perspective)
    print('asking it to foo(14)')
    perspective.callRemote('foo', 14)
main()
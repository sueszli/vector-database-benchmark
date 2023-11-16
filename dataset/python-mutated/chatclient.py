from twisted.cred import credentials
from twisted.internet import reactor
from twisted.spread import pb

class Client(pb.Referenceable):

    def remote_print(self, message):
        if False:
            return 10
        print(message)

    def connect(self):
        if False:
            while True:
                i = 10
        factory = pb.PBClientFactory()
        reactor.connectTCP('localhost', 8800, factory)
        def1 = factory.login(credentials.UsernamePassword('alice', '1234'), client=self)
        def1.addCallback(self.connected)
        reactor.run()

    def connected(self, perspective):
        if False:
            i = 10
            return i + 15
        print('connected, joining group #NeedAFourth')
        self.perspective = perspective
        d = perspective.callRemote('joinGroup', '#NeedAFourth')
        d.addCallback(self.gotGroup)

    def gotGroup(self, group):
        if False:
            while True:
                i = 10
        print('joined group, now sending a message to all members')
        d = group.callRemote('send', 'You can call me Al.')
        d.addCallback(self.shutdown)

    def shutdown(self, result):
        if False:
            print('Hello World!')
        reactor.stop()
Client().connect()
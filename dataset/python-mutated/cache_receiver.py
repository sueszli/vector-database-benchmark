import cache_classes
from twisted.application import internet, service
from twisted.internet import reactor
from twisted.spread import pb

class Receiver(pb.Root):

    def remote_takePond(self, pond):
        if False:
            while True:
                i = 10
        self.pond = pond
        print('got pond:', pond)
        self.remote_checkDucks()

    def remote_checkDucks(self):
        if False:
            for i in range(10):
                print('nop')
        print('[%d] ducks: ' % self.pond.count(), self.pond.getDucks())

    def remote_ignorePond(self):
        if False:
            print('Hello World!')
        print('dropping pond')
        self.pond = None

    def remote_shutdown(self):
        if False:
            print('Hello World!')
        reactor.stop()
application = service.Application('copy_receiver')
internet.TCPServer(8800, pb.PBServerFactory(Receiver())).setServiceParent(service.IServiceCollection(application))
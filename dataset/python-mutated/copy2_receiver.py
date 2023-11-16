import copy2_classes
from twisted.application import internet, service
from twisted.internet import reactor
from twisted.spread import pb

class Receiver(pb.Root):

    def remote_takePond(self, pond):
        if False:
            print('Hello World!')
        print(' got pond:', pond)
        print(' count %d' % pond.count())
        return 'safe and sound'

    def remote_shutdown(self):
        if False:
            while True:
                i = 10
        reactor.stop()
application = service.Application('copy_receiver')
internet.TCPServer(8800, pb.PBServerFactory(Receiver())).setServiceParent(service.IServiceCollection(application))
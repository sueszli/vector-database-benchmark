from twisted.internet import reactor
from twisted.spread import pb

class One(pb.Root):

    def remote_takeTwo(self, two):
        if False:
            while True:
                i = 10
        print('received a Two called', two)
        print('telling it to print(12)')
        two.callRemote('print', 12)
reactor.listenTCP(8800, pb.PBServerFactory(One()))
reactor.run()
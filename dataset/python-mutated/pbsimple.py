from twisted.internet import reactor
from twisted.spread import pb

class Echoer(pb.Root):

    def remote_echo(self, st):
        if False:
            return 10
        print('echoing:', st)
        return st
if __name__ == '__main__':
    reactor.listenTCP(8789, pb.PBServerFactory(Echoer()))
    reactor.run()
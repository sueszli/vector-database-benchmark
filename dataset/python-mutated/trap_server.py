from twisted.internet import reactor
from twisted.spread import pb

class MyException(pb.Error):
    pass

class One(pb.Root):

    def remote_fooMethod(self, arg):
        if False:
            print('Hello World!')
        if arg == 'panic!':
            raise MyException
        return 'response'

    def remote_shutdown(self):
        if False:
            i = 10
            return i + 15
        reactor.stop()
reactor.listenTCP(8800, pb.PBServerFactory(One()))
reactor.run()
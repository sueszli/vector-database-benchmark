from copy2_classes import SenderPond
from twisted.internet import reactor
from twisted.python import log
from twisted.spread import jelly, pb

class Sender:

    def __init__(self, pond):
        if False:
            print('Hello World!')
        self.pond = pond

    def got_obj(self, obj):
        if False:
            print('Hello World!')
        d = obj.callRemote('takePond', self.pond)
        d.addCallback(self.ok).addErrback(self.notOk)

    def ok(self, response):
        if False:
            i = 10
            return i + 15
        print('pond arrived', response)
        reactor.stop()

    def notOk(self, failure):
        if False:
            print('Hello World!')
        print('error during takePond:')
        if failure.type == jelly.InsecureJelly:
            print(' InsecureJelly')
        else:
            print(failure)
        reactor.stop()
        return None

def main():
    if False:
        print('Hello World!')
    pond = SenderPond(3, 4)
    print('count %d' % pond.count())
    sender = Sender(pond)
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    deferred = factory.getRootObject()
    deferred.addCallback(sender.got_obj)
    reactor.run()
if __name__ == '__main__':
    main()
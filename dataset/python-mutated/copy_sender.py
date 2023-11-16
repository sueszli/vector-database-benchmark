from twisted.internet import reactor
from twisted.python import log
from twisted.spread import jelly, pb

class LilyPond:

    def setStuff(self, color, numFrogs):
        if False:
            i = 10
            return i + 15
        self.color = color
        self.numFrogs = numFrogs

    def countFrogs(self):
        if False:
            while True:
                i = 10
        print('%d frogs' % self.numFrogs)

class CopyPond(LilyPond, pb.Copyable):
    pass

class Sender:

    def __init__(self, pond):
        if False:
            while True:
                i = 10
        self.pond = pond

    def got_obj(self, remote):
        if False:
            for i in range(10):
                print('nop')
        self.remote = remote
        d = remote.callRemote('takePond', self.pond)
        d.addCallback(self.ok).addErrback(self.notOk)

    def ok(self, response):
        if False:
            return 10
        print('pond arrived', response)
        reactor.stop()

    def notOk(self, failure):
        if False:
            return 10
        print('error during takePond:')
        if failure.type == jelly.InsecureJelly:
            print(' InsecureJelly')
        else:
            print(failure)
        reactor.stop()
        return None

def main():
    if False:
        return 10
    from copy_sender import CopyPond
    pond = CopyPond()
    pond.setStuff('green', 7)
    pond.countFrogs()
    print('.'.join([pond.__class__.__module__, pond.__class__.__name__]))
    sender = Sender(pond)
    factory = pb.PBClientFactory()
    reactor.connectTCP('localhost', 8800, factory)
    deferred = factory.getRootObject()
    deferred.addCallback(sender.got_obj)
    reactor.run()
if __name__ == '__main__':
    main()
import time
from twisted.cred.credentials import UsernamePassword
from twisted.internet import defer, reactor
from twisted.spread import pb

class PBBenchClient:
    hostname = 'localhost'
    portno = pb.portno
    calledThisSecond = 0

    def callLoop(self, ignored):
        if False:
            for i in range(10):
                print('nop')
        d1 = self.persp.callRemote(b'simple')
        d2 = self.persp.callRemote(b'complexTypes')
        defer.DeferredList([d1, d2]).addCallback(self.callLoop)
        self.calledThisSecond += 1
        thisSecond = int(time.time())
        if thisSecond != self.lastSecond:
            if thisSecond - self.lastSecond > 1:
                print('WARNING it took more than one second')
            print('cps:', self.calledThisSecond)
            self.calledThisSecond = 0
            self.lastSecond = thisSecond

    def _cbPerspective(self, persp):
        if False:
            print('Hello World!')
        self.persp = persp
        self.lastSecond = int(time.time())
        self.callLoop(None)

    def runTest(self):
        if False:
            print('Hello World!')
        factory = pb.PBClientFactory()
        reactor.connectTCP(self.hostname, self.portno, factory)
        factory.login(UsernamePassword(b'benchmark', b'benchmark')).addCallback(self._cbPerspective)

def main():
    if False:
        while True:
            i = 10
    PBBenchClient().runTest()
    from twisted.internet import reactor
    reactor.run()
if __name__ == '__main__':
    main()
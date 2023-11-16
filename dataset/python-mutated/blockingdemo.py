from twisted.internet import _threadedselect
_threadedselect.install()
from itertools import count
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure
from twisted.python.runtime import seconds
try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue

class TwistedManager:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.twistedQueue = Queue()
        self.key = count()
        self.results = {}

    def getKey(self):
        if False:
            return 10
        return next(self.key)

    def start(self):
        if False:
            print('Hello World!')
        reactor.interleave(self.twistedQueue.put)

    def _stopIterating(self, value, key):
        if False:
            i = 10
            return i + 15
        self.results[key] = value

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        key = self.getKey()
        reactor.addSystemEventTrigger('after', 'shutdown', self._stopIterating, True, key)
        reactor.stop()
        self.iterate(key)

    def getDeferred(self, d):
        if False:
            for i in range(10):
                print('nop')
        key = self.getKey()
        d.addBoth(self._stopIterating, key)
        res = self.iterate(key)
        if isinstance(res, Failure):
            res.raiseException()
        return res

    def poll(self, noLongerThan=1.0):
        if False:
            print('Hello World!')
        base = seconds()
        try:
            while seconds() - base <= noLongerThan:
                callback = self.twistedQueue.get_nowait()
                callback()
        except Empty:
            pass

    def iterate(self, key=None):
        if False:
            return 10
        while key not in self.results:
            callback = self.twistedQueue.get()
            callback()
        return self.results.pop(key)

def fakeDeferred(msg):
    if False:
        while True:
            i = 10
    d = Deferred()

    def cb():
        if False:
            while True:
                i = 10
        print('deferred called back')
        d.callback(msg)
    reactor.callLater(2, cb)
    return d

def fakeCallback():
    if False:
        return 10
    print('twisted is still running')

def main():
    if False:
        for i in range(10):
            print('nop')
    m = TwistedManager()
    print('starting')
    m.start()
    print('setting up a 1sec callback')
    reactor.callLater(1, fakeCallback)
    print('getting a deferred')
    res = m.getDeferred(fakeDeferred('got it!'))
    print('got the deferred:', res)
    print('stopping')
    m.stop()
    print('stopped')
if __name__ == '__main__':
    main()
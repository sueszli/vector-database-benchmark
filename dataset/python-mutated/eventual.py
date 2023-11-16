from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import log

class _SimpleCallQueue:
    _reactor = reactor

    def __init__(self):
        if False:
            while True:
                i = 10
        self._events = []
        self._flushObservers = []
        self._timer = None
        self._in_turn = False

    def append(self, cb, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._events.append((cb, args, kwargs))
        if not self._timer:
            self._timer = self._reactor.callLater(0, self._turn)

    def _turn(self):
        if False:
            for i in range(10):
                print('nop')
        self._timer = None
        self._in_turn = True
        (events, self._events) = (self._events, [])
        for (cb, args, kwargs) in events:
            try:
                cb(*args, **kwargs)
            except Exception:
                log.err()
        self._in_turn = False
        if self._events and (not self._timer):
            self._timer = self._reactor.callLater(0, self._turn)
        if not self._events:
            (observers, self._flushObservers) = (self._flushObservers, [])
            for o in observers:
                o.callback(None)

    def flush(self):
        if False:
            while True:
                i = 10
        if not self._events and (not self._in_turn):
            return defer.succeed(None)
        d = defer.Deferred()
        self._flushObservers.append(d)
        return d
_theSimpleQueue = _SimpleCallQueue()

def eventually(cb, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    _theSimpleQueue.append(cb, args, kwargs)

def fireEventually(value=None):
    if False:
        for i in range(10):
            print('nop')
    d = defer.Deferred()
    eventually(d.callback, value)
    return d

def flushEventualQueue(_ignored=None):
    if False:
        return 10
    return _theSimpleQueue.flush()

def _setReactor(r=None):
    if False:
        print('Hello World!')
    if r is None:
        r = reactor
    _theSimpleQueue._reactor = r
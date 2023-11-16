import weakref
from gevent.queue import Queue

class BroadcastQueue:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._queues = []

    def register(self):
        if False:
            for i in range(10):
                print('nop')
        q = Queue()
        self._queues.append(weakref.ref(q))
        return q

    def broadcast(self, val):
        if False:
            print('Hello World!')
        for q in list(self._queues):
            if q():
                q().put(val)
            else:
                self._queues.remove(q)
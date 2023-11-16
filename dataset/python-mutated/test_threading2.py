from collections import deque
from panda3d import core
from direct.stdpy import threading2

def test_threading2():
    if False:
        print('Hello World!')

    class BoundedQueue(threading2._Verbose):

        def __init__(self, limit):
            if False:
                for i in range(10):
                    print('nop')
            threading2._Verbose.__init__(self)
            self.mon = threading2.RLock()
            self.rc = threading2.Condition(self.mon)
            self.wc = threading2.Condition(self.mon)
            self.limit = limit
            self.queue = deque()

        def put(self, item):
            if False:
                return 10
            self.mon.acquire()
            while len(self.queue) >= self.limit:
                self._note('put(%s): queue full', item)
                self.wc.wait()
            self.queue.append(item)
            self._note('put(%s): appended, length now %d', item, len(self.queue))
            self.rc.notify()
            self.mon.release()

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            self.mon.acquire()
            while not self.queue:
                self._note('get(): queue empty')
                self.rc.wait()
            item = self.queue.popleft()
            self._note('get(): got %s, %d left', item, len(self.queue))
            self.wc.notify()
            self.mon.release()
            return item

    class ProducerThread(threading2.Thread):

        def __init__(self, queue, quota):
            if False:
                for i in range(10):
                    print('nop')
            threading2.Thread.__init__(self, name='Producer')
            self.queue = queue
            self.quota = quota

        def run(self):
            if False:
                while True:
                    i = 10
            from random import random
            counter = 0
            while counter < self.quota:
                counter = counter + 1
                self.queue.put('%s.%d' % (self.getName(), counter))
                core.Thread.sleep(random() * 1e-05)

    class ConsumerThread(threading2.Thread):

        def __init__(self, queue, count):
            if False:
                return 10
            threading2.Thread.__init__(self, name='Consumer')
            self.queue = queue
            self.count = count

        def run(self):
            if False:
                while True:
                    i = 10
            while self.count > 0:
                item = self.queue.get()
                print(item)
                self.count = self.count - 1
    NP = 3
    QL = 4
    NI = 5
    Q = BoundedQueue(QL)
    P = []
    for i in range(NP):
        t = ProducerThread(Q, NI)
        t.setName('Producer-%d' % (i + 1))
        P.append(t)
    C = ConsumerThread(Q, NI * NP)
    for t in P:
        t.start()
        core.Thread.sleep(1e-06)
    C.start()
    for t in P:
        t.join()
    C.join()
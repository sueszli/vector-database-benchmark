import threading
import collections

class BoundedBlockingQueue(object):

    def __init__(self, capacity):
        if False:
            while True:
                i = 10
        '\n        :type capacity: int\n        '
        self.__cv = threading.Condition()
        self.__q = collections.deque()
        self.__cap = capacity

    def enqueue(self, element):
        if False:
            print('Hello World!')
        '\n        :type element: int\n        :rtype: void\n        '
        with self.__cv:
            while len(self.__q) == self.__cap:
                self.__cv.wait()
            self.__q.append(element)
            self.__cv.notifyAll()

    def dequeue(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        with self.__cv:
            while not self.__q:
                self.__cv.wait()
            self.__cv.notifyAll()
            return self.__q.popleft()

    def size(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        with self.__cv:
            return len(self.__q)
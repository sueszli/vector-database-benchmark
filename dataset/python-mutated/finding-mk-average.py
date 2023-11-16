import collections
from sortedcontainers import SortedList

class MKAverage(object):

    def __init__(self, m, k):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        :type k: int\n        '
        self.__m = m
        self.__k = k
        self.__dq = collections.deque()
        self.__sl = SortedList()
        self.__total = self.__first_k = self.__last_k = 0

    def addElement(self, num):
        if False:
            print('Hello World!')
        '\n        :type num: int\n        :rtype: None\n        '
        if len(self.__dq) == self.__m:
            self.__remove(self.__dq.popleft())
        self.__dq.append(num)
        self.__add(num)

    def calculateMKAverage(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: int\n        '
        if len(self.__sl) < self.__m:
            return -1
        return (self.__total - self.__first_k - self.__last_k) // (self.__m - 2 * self.__k)

    def __add(self, num):
        if False:
            for i in range(10):
                print('nop')
        self.__total += num
        idx = self.__sl.bisect_left(num)
        if idx < self.__k:
            self.__first_k += num
            if len(self.__sl) >= self.__k:
                self.__first_k -= self.__sl[self.__k - 1]
        if idx > len(self.__sl) - self.__k:
            self.__last_k += num
            if len(self.__sl) >= self.__k:
                self.__last_k -= self.__sl[-self.__k]
        self.__sl.add(num)

    def __remove(self, num):
        if False:
            print('Hello World!')
        self.__total -= num
        idx = self.__sl.index(num)
        if idx < self.__k:
            self.__first_k -= num
            self.__first_k += self.__sl[self.__k]
        elif idx > len(self.__sl) - 1 - self.__k:
            self.__last_k -= num
            self.__last_k += self.__sl[-1 - self.__k]
        self.__sl.remove(num)
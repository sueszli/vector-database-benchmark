from collections import deque

class MovingAverage(object):

    def __init__(self, size):
        if False:
            while True:
                i = 10
        '\n        Initialize your data structure here.\n        :type size: int\n        '
        self.__size = size
        self.__sum = 0
        self.__q = deque()

    def next(self, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type val: int\n        :rtype: float\n        '
        if len(self.__q) == self.__size:
            self.__sum -= self.__q.popleft()
        self.__sum += val
        self.__q.append(val)
        return 1.0 * self.__sum / len(self.__q)
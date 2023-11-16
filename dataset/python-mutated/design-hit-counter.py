from collections import deque

class HitCounter(object):

    def __init__(self):
        if False:
            return 10
        '\n        Initialize your data structure here.\n        '
        self.__k = 300
        self.__dq = deque()
        self.__count = 0

    def hit(self, timestamp):
        if False:
            print('Hello World!')
        '\n        Record a hit.\n        @param timestamp - The current timestamp (in seconds granularity).\n        :type timestamp: int\n        :rtype: void\n        '
        self.getHits(timestamp)
        if self.__dq and self.__dq[-1][0] == timestamp:
            self.__dq[-1][1] += 1
        else:
            self.__dq.append([timestamp, 1])
        self.__count += 1

    def getHits(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of hits in the past 5 minutes.\n        @param timestamp - The current timestamp (in seconds granularity).\n        :type timestamp: int\n        :rtype: int\n        '
        while self.__dq and self.__dq[0][0] <= timestamp - self.__k:
            self.__count -= self.__dq.popleft()[1]
        return self.__count
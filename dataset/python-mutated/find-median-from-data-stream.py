from heapq import heappush, heappop

class MedianFinder(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Initialize your data structure here.\n        '
        self.__max_heap = []
        self.__min_heap = []

    def addNum(self, num):
        if False:
            i = 10
            return i + 15
        '\n        Adds a num into the data structure.\n        :type num: int\n        :rtype: void\n        '
        if not self.__max_heap or num > -self.__max_heap[0]:
            heappush(self.__min_heap, num)
            if len(self.__min_heap) > len(self.__max_heap) + 1:
                heappush(self.__max_heap, -heappop(self.__min_heap))
        else:
            heappush(self.__max_heap, -num)
            if len(self.__max_heap) > len(self.__min_heap):
                heappush(self.__min_heap, -heappop(self.__max_heap))

    def findMedian(self):
        if False:
            print('Hello World!')
        '\n        Returns the median of current data stream\n        :rtype: float\n        '
        return (-self.__max_heap[0] + self.__min_heap[0]) / 2.0 if len(self.__min_heap) == len(self.__max_heap) else self.__min_heap[0]
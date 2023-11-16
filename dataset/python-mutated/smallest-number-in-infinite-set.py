import heapq

class SmallestInfiniteSet(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__n = 1
        self.__lookup = set()
        self.__min_heap = []

    def popSmallest(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        if self.__min_heap:
            result = heapq.heappop(self.__min_heap)
            self.__lookup.remove(result)
            return result
        result = self.__n
        self.__n += 1
        return result

    def addBack(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: None\n        '
        if num >= self.__n or num in self.__lookup:
            return
        self.__lookup.add(num)
        heapq.heappush(self.__min_heap, num)
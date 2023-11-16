import heapq

class DinnerPlates(object):

    def __init__(self, capacity):
        if False:
            return 10
        '\n        :type capacity: int\n        '
        self.__stks = []
        self.__c = capacity
        self.__min_heap = []

    def push(self, val):
        if False:
            i = 10
            return i + 15
        '\n        :type val: int\n        :rtype: None\n        '
        if self.__min_heap:
            l = heapq.heappop(self.__min_heap)
            if l < len(self.__stks):
                self.__stks[l].append(val)
                return
            self.__min_heap = []
        if not self.__stks or len(self.__stks[-1]) == self.__c:
            self.__stks.append([])
        self.__stks[-1].append(val)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: int\n        '
        while self.__stks and (not self.__stks[-1]):
            self.__stks.pop()
        if not self.__stks:
            return -1
        return self.__stks[-1].pop()

    def popAtStack(self, index):
        if False:
            i = 10
            return i + 15
        '\n        :type index: int\n        :rtype: int\n        '
        if index >= len(self.__stks) or not self.__stks[index]:
            return -1
        heapq.heappush(self.__min_heap, index)
        return self.__stks[index].pop()
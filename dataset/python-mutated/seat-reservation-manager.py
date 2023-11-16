import heapq

class SeatManager(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        '
        self.__min_heap = range(1, n + 1)

    def reserve(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        return heapq.heappop(self.__min_heap)

    def unreserve(self, seatNumber):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type seatNumber: int\n        :rtype: None\n        '
        heapq.heappush(self.__min_heap, seatNumber)
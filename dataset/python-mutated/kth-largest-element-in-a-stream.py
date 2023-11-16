import heapq

class KthLargest(object):

    def __init__(self, k, nums):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :type nums: List[int]\n        '
        self.__k = k
        self.__min_heap = []
        for n in nums:
            self.add(n)

    def add(self, val):
        if False:
            while True:
                i = 10
        '\n        :type val: int\n        :rtype: int\n        '
        heapq.heappush(self.__min_heap, val)
        if len(self.__min_heap) > self.__k:
            heapq.heappop(self.__min_heap)
        return self.__min_heap[0]
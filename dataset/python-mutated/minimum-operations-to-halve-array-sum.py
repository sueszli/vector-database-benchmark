import heapq

class Solution(object):

    def halveArray(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        target = sum(nums) / 2.0
        max_heap = [-x for x in nums]
        heapq.heapify(max_heap)
        result = 1
        while max_heap:
            x = -heapq.heappop(max_heap) / 2.0
            target -= x
            if target <= 0.0:
                break
            heapq.heappush(max_heap, -x)
            result += 1
        return result
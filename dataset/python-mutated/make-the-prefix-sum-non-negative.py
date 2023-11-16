import heapq

class Solution(object):

    def makePrefSumNonNegative(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = prefix = 0
        min_heap = []
        for x in nums:
            heapq.heappush(min_heap, x)
            prefix += x
            if prefix < 0:
                prefix -= heapq.heappop(min_heap)
                result += 1
        return result
import itertools
import heapq

class Solution(object):

    def maxScore(self, nums1, nums2, k):
        if False:
            while True:
                i = 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type k: int\n        :rtype: int\n        '
        result = curr = 0
        min_heap = []
        for (a, b) in sorted(itertools.izip(nums1, nums2), key=lambda x: x[1], reverse=True):
            curr += a
            heapq.heappush(min_heap, a)
            if len(min_heap) > k:
                curr -= heapq.heappop(min_heap)
            if len(min_heap) == k:
                result = max(result, curr * b)
        return result
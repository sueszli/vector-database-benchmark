from heapq import heappush, heappop

class Solution(object):

    def kSmallestPairs(self, nums1, nums2, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type k: int\n        :rtype: List[List[int]]\n        '
        pairs = []
        if len(nums1) > len(nums2):
            tmp = self.kSmallestPairs(nums2, nums1, k)
            for pair in tmp:
                pairs.append([pair[1], pair[0]])
            return pairs
        min_heap = []

        def push(i, j):
            if False:
                for i in range(10):
                    print('nop')
            if i < len(nums1) and j < len(nums2):
                heappush(min_heap, [nums1[i] + nums2[j], i, j])
        push(0, 0)
        while min_heap and len(pairs) < k:
            (_, i, j) = heappop(min_heap)
            pairs.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
        return pairs
from heapq import nsmallest
from itertools import product

class Solution2(object):

    def kSmallestPairs(self, nums1, nums2, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type k: int\n        :rtype: List[List[int]]\n        '
        return nsmallest(k, product(nums1, nums2), key=sum)
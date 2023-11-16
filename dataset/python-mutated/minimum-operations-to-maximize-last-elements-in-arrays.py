import itertools

class Solution(object):

    def minOperations(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        cnt = [0] * 2
        for (x, y) in itertools.izip(nums1, nums2):
            if not (min(x, y) <= min(nums1[-1], nums2[-1]) and max(x, y) <= max(nums1[-1], nums2[-1])):
                return -1
            if not (x <= nums1[-1] and y <= nums2[-1]):
                cnt[0] += 1
            if not (x <= nums2[-1] and y <= nums1[-1]):
                cnt[1] += 1
        return min(cnt)
import itertools

class Solution2(object):

    def minOperations(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        INF = float('inf')

        def count(mx1, mx2):
            if False:
                i = 10
                return i + 15
            return sum((1 if y <= mx1 and x <= mx2 else INF for (x, y) in itertools.izip(nums1, nums2) if not (x <= mx1 and y <= mx2)))
        result = min(count(nums1[-1], nums2[-1]), count(nums2[-1], nums1[-1]))
        return result if result != INF else -1
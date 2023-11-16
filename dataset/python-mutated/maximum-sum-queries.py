import itertools
import bisect

class Solution(object):

    def maximumSumQueries(self, nums1, nums2, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        pairs = sorted(((i, j) for (i, j) in itertools.izip(nums1, nums2)))
        result = [0] * len(queries)
        stk = []
        for (x, y, i) in sorted(((x, y, i) for (i, (x, y)) in enumerate(queries)), reverse=True):
            while pairs and pairs[-1][0] >= x:
                (a, b) = pairs.pop()
                while stk and stk[-1][1] <= a + b:
                    stk.pop()
                if not stk or stk[-1][0] < b:
                    stk.append((b, a + b))
            j = bisect.bisect_left(stk, (y,))
            result[i] = stk[j][1] if j != len(stk) else -1
        return result
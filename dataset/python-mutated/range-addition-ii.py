class Solution(object):

    def maxCount(self, m, n, ops):
        if False:
            return 10
        '\n        :type m: int\n        :type n: int\n        :type ops: List[List[int]]\n        :rtype: int\n        '
        for op in ops:
            m = min(m, op[0])
            n = min(n, op[1])
        return m * n
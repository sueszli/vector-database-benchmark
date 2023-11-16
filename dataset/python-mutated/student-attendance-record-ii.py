class Solution(object):

    def checkRecord(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        M = 1000000007
        (a0l0, a0l1, a0l2, a1l0, a1l1, a1l2) = (1, 0, 0, 0, 0, 0)
        for i in xrange(n + 1):
            (a0l2, a0l1, a0l0) = (a0l1, a0l0, (a0l0 + a0l1 + a0l2) % M)
            (a1l2, a1l1, a1l0) = (a1l1, a1l0, (a0l0 + a1l0 + a1l1 + a1l2) % M)
        return a1l0
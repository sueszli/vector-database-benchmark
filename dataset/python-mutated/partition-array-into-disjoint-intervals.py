class Solution(object):

    def partitionDisjoint(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[int]\n        :rtype: int\n        '
        B = A[:]
        for i in reversed(xrange(len(A) - 1)):
            B[i] = min(B[i], B[i + 1])
        p_max = 0
        for i in xrange(1, len(A)):
            p_max = max(p_max, A[i - 1])
            if p_max <= B[i]:
                return i
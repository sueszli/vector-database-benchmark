class Solution(object):

    def smallestRepunitDivByK(self, K):
        if False:
            while True:
                i = 10
        '\n        :type K: int\n        :rtype: int\n        '
        if K % 2 == 0 or K % 5 == 0:
            return -1
        result = 0
        for N in xrange(1, K + 1):
            result = (result * 10 + 1) % K
            if not result:
                return N
        assert False
        return -1
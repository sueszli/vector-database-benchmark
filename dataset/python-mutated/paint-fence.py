class Solution(object):

    def numWays(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        if n == 0:
            return 0
        elif n == 1:
            return k
        ways = [0] * 3
        ways[0] = k
        ways[1] = (k - 1) * ways[0] + k
        for i in xrange(2, n):
            ways[i % 3] = (k - 1) * (ways[(i - 1) % 3] + ways[(i - 2) % 3])
        return ways[(n - 1) % 3]

class Solution2(object):

    def numWays(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        if n == 0:
            return 0
        elif n == 1:
            return k
        ways = [0] * n
        ways[0] = k
        ways[1] = (k - 1) * ways[0] + k
        for i in xrange(2, n):
            ways[i] = (k - 1) * (ways[i - 1] + ways[i - 2])
        return ways[n - 1]
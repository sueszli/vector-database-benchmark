class Solution(object):

    def numsSameConsecDiff(self, N, K):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :type K: int\n        :rtype: List[int]\n        '
        curr = range(10)
        for i in xrange(N - 1):
            curr = [x * 10 + y for x in curr for y in set([x % 10 + K, x % 10 - K]) if x and 0 <= y < 10]
        return curr
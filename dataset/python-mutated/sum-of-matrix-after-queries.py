class Solution(object):

    def matrixSumQueries(self, n, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type queries: List[List[int]]\n        :rtype: int\n        '
        lookup = [[False] * n for _ in xrange(2)]
        cnt = [0] * 2
        result = 0
        for (t, i, v) in reversed(queries):
            if lookup[t][i]:
                continue
            lookup[t][i] = True
            cnt[t] += 1
            result += v * (n - cnt[t ^ 1])
        return result
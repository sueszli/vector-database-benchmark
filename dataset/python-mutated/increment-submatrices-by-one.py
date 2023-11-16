class Solution(object):

    def rangeAddQueries(self, n, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type queries: List[List[int]]\n        :rtype: List[List[int]]\n        '
        result = [[0] * n for _ in xrange(n)]
        for (r1, c1, r2, c2) in queries:
            result[r1][c1] += 1
            if c2 + 1 < len(result[0]):
                result[r1][c2 + 1] -= 1
            if r2 + 1 < len(result):
                result[r2 + 1][c1] -= 1
            if r2 + 1 < len(result) and c2 + 1 < len(result[0]):
                result[r2 + 1][c2 + 1] += 1
        for r in xrange(len(result)):
            for c in xrange(len(result[0]) - 1):
                result[r][c + 1] += result[r][c]
        for r in xrange(len(result) - 1):
            for c in xrange(len(result[0])):
                result[r + 1][c] += result[r][c]
        return result
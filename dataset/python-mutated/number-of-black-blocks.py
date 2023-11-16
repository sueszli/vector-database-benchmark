import collections

class Solution(object):

    def countBlackBlocks(self, m, n, coordinates):
        if False:
            print('Hello World!')
        '\n        :type m: int\n        :type n: int\n        :type coordinates: List[List[int]]\n        :rtype: List[int]\n        '
        L = 2
        cnt = collections.Counter()
        for (x, y) in coordinates:
            for nx in xrange(max(x - (L - 1), 0), min(x + 1, m - (L - 1))):
                for ny in xrange(max(y - (L - 1), 0), min(y + 1, n - (L - 1))):
                    cnt[nx, ny] += 1
        result = [0] * (L ** 2 + 1)
        for c in cnt.itervalues():
            result[c] += 1
        result[0] = (m - (L - 1)) * (n - (L - 1)) - sum((result[i] for i in xrange(1, len(result))))
        return result
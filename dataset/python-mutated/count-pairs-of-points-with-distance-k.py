import collections

class Solution(object):

    def countPairs(self, coordinates, k):
        if False:
            i = 10
            return i + 15
        '\n        :type coordinates: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        result = 0
        cnt = collections.Counter()
        for (x, y) in coordinates:
            for i in xrange(k + 1):
                result += cnt.get((x ^ i, y ^ k - i), 0)
            cnt[x, y] += 1
        return result
class Solution(object):

    def circularGameLosers(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: List[int]\n        '
        lookup = [False] * n
        idx = 0
        for i in xrange(n):
            if lookup[idx]:
                break
            lookup[idx] = True
            idx = (idx + (i + 1) * k) % n
        return [i + 1 for i in xrange(n) if not lookup[i]]
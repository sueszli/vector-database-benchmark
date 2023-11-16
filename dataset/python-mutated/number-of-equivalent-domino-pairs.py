import collections

class Solution(object):

    def numEquivDominoPairs(self, dominoes):
        if False:
            return 10
        '\n        :type dominoes: List[List[int]]\n        :rtype: int\n        '
        counter = collections.Counter(((min(x), max(x)) for x in dominoes))
        return sum((v * (v - 1) // 2 for v in counter.itervalues()))
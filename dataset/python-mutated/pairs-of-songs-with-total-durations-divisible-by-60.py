import collections

class Solution(object):

    def numPairsDivisibleBy60(self, time):
        if False:
            return 10
        '\n        :type time: List[int]\n        :rtype: int\n        '
        result = 0
        count = collections.Counter()
        for t in time:
            result += count[-t % 60]
            count[t % 60] += 1
        return result
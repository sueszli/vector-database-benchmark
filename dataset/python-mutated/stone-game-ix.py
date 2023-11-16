import collections

class Solution(object):

    def stoneGameIX(self, stones):
        if False:
            i = 10
            return i + 15
        '\n        :type stones: List[int]\n        :rtype: bool\n        '
        count = collections.Counter((x % 3 for x in stones))
        if count[0] % 2 == 0:
            return count[1] and count[2]
        return abs(count[1] - count[2]) >= 3
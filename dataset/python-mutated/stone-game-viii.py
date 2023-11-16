class Solution(object):

    def stoneGameVIII(self, stones):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type stones: List[int]\n        :rtype: int\n        '
        for i in xrange(len(stones) - 1):
            stones[i + 1] += stones[i]
        return reduce(lambda curr, i: max(curr, stones[i] - curr), reversed(xrange(1, len(stones) - 1)), stones[-1])
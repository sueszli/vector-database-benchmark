class Solution(object):

    def maxJump(self, stones):
        if False:
            print('Hello World!')
        '\n        :type stones: List[int]\n        :rtype: int\n        '
        return stones[1] - stones[0] if len(stones) == 2 else max((stones[i + 2] - stones[i] for i in xrange(len(stones) - 2)))
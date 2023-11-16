class Solution(object):

    def stoneGame(self, piles):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type piles: List[int]\n        :rtype: bool\n        '
        if len(piles) % 2 == 0 or len(piles) == 1:
            return True
        dp = [0] * len(piles)
        for i in reversed(xrange(len(piles))):
            dp[i] = piles[i]
            for j in xrange(i + 1, len(piles)):
                dp[j] = max(piles[i] - dp[j], piles[j] - dp[j - 1])
        return dp[-1] >= 0
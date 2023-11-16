class Solution(object):

    def maxValueOfCoins(self, piles, k):
        if False:
            print('Hello World!')
        '\n        :type piles: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        dp = [0]
        for pile in piles:
            new_dp = [0] * min(len(dp) + len(pile), k + 1)
            for i in xrange(len(dp)):
                curr = 0
                for j in xrange(min(k - i, len(pile)) + 1):
                    new_dp[i + j] = max(new_dp[i + j], dp[i] + curr)
                    curr += pile[j] if j < len(pile) else 0
            dp = new_dp
        return dp[-1]
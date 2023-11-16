class Solution(object):

    def maximizeWin(self, prizePositions, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prizePositions: List[int]\n        :type k: int\n        :rtype: int\n        '
        dp = [0] * (len(prizePositions) + 1)
        result = left = 0
        for right in xrange(len(prizePositions)):
            while prizePositions[right] - prizePositions[left] > k:
                left += 1
            dp[right + 1] = max(dp[right], right - left + 1)
            result = max(result, dp[left] + (right - left + 1))
        return result
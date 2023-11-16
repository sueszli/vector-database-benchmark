class Solution(object):

    def minimumWhiteTiles(self, floor, numCarpets, carpetLen):
        if False:
            i = 10
            return i + 15
        '\n        :type floor: str\n        :type numCarpets: int\n        :type carpetLen: int\n        :rtype: int\n        '
        dp = [[0] * (numCarpets + 1) for _ in xrange(len(floor) + 1)]
        for i in xrange(1, len(dp)):
            dp[i][0] = dp[i - 1][0] + int(floor[i - 1])
            for j in xrange(1, numCarpets + 1):
                dp[i][j] = min(dp[i - 1][j] + int(floor[i - 1]), dp[max(i - carpetLen, 0)][j - 1])
        return dp[-1][-1]
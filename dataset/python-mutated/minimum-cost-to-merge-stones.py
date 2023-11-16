class Solution(object):

    def mergeStones(self, stones, K):
        if False:
            return 10
        '\n        :type stones: List[int]\n        :type K: int\n        :rtype: int\n        '
        if (len(stones) - 1) % (K - 1):
            return -1
        prefix = [0]
        for x in stones:
            prefix.append(prefix[-1] + x)
        dp = [[0] * len(stones) for _ in xrange(len(stones))]
        for l in xrange(K - 1, len(stones)):
            for i in xrange(len(stones) - l):
                dp[i][i + l] = min((dp[i][j] + dp[j + 1][i + l] for j in xrange(i, i + l, K - 1)))
                if l % (K - 1) == 0:
                    dp[i][i + l] += prefix[i + l + 1] - prefix[i]
        return dp[0][len(stones) - 1]
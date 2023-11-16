class Solution(object):

    def stoneGameVII(self, stones):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type stones: List[int]\n        :rtype: int\n        '

        def score(i, j):
            if False:
                i = 10
                return i + 15
            return prefix[j + 1] - prefix[i]
        prefix = [0]
        for stone in stones:
            prefix.append(prefix[-1] + stone)
        dp = [[0 for _ in xrange(len(stones))] for _ in xrange(2)]
        for i in reversed(xrange(len(stones))):
            for j in xrange(i + 1, len(stones)):
                dp[i % 2][j] = max(score(i + 1, j) - dp[(i + 1) % 2][j], score(i, j - 1) - dp[i % 2][j - 1])
        return dp[0][-1]
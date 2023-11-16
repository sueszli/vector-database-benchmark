class Solution(object):

    def numberOfCombinations(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def find_longest_common_prefix(num):
            if False:
                i = 10
                return i + 15
            lcp = [[0] * (len(num) + 1) for _ in xrange(len(num) + 1)]
            for i in reversed(xrange(len(lcp) - 1)):
                for j in reversed(xrange(len(lcp[0]) - 1)):
                    if num[i] == num[j]:
                        lcp[i][j] = lcp[i + 1][j + 1] + 1
            return lcp

        def is_less_or_equal_to_with_same_length(num, lcp, i, j, l):
            if False:
                print('Hello World!')
            return lcp[i][j] >= l or num[i + lcp[i][j]] < num[j + lcp[i][j]]
        lcp = find_longest_common_prefix(num)
        dp = [[0] * len(num) for _ in xrange(len(num))]
        dp[0][0] = int(num[0] != '0')
        for i in xrange(1, len(num)):
            dp[i][i] = dp[i - 1][i - 1]
            if num[i] == '0':
                continue
            accu = 0
            for l in xrange(len(num) - i + 1):
                ni = i + l - 1
                dp[ni][l - 1] = accu
                if i - l < 0:
                    continue
                if num[i - l] != '0' and is_less_or_equal_to_with_same_length(num, lcp, i - l, i, l):
                    dp[ni][l - 1] = (dp[ni][l - 1] + dp[i - 1][l - 1]) % MOD
                accu = (accu + dp[i - 1][l - 1]) % MOD
        return reduce(lambda total, x: (total + x) % MOD, dp[-1], 0)
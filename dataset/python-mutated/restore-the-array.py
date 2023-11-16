class Solution(object):

    def numberOfArrays(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        klen = len(str(k))
        dp = [0] * (klen + 1)
        dp[len(s) % len(dp)] = 1
        for i in reversed(xrange(len(s))):
            dp[i % len(dp)] = 0
            if s[i] == '0':
                continue
            curr = 0
            for j in xrange(i, min(i + klen, len(s))):
                curr = 10 * curr + int(s[j])
                if curr > k:
                    break
                dp[i % len(dp)] = (dp[i % len(dp)] + dp[(j + 1) % len(dp)]) % MOD
        return dp[0]
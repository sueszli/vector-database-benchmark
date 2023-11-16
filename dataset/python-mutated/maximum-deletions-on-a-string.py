class Solution(object):

    def deleteString(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        if all((x == s[0] for x in s)):
            return len(s)
        dp2 = [[0] * (len(s) + 1) for i in xrange(2)]
        dp = [1] * len(s)
        for i in reversed(xrange(len(s) - 1)):
            for j in xrange(i + 1, len(s)):
                dp2[i % 2][j] = dp2[(i + 1) % 2][j + 1] + 1 if s[j] == s[i] else 0
                if dp2[i % 2][j] >= j - i:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[0]

class Solution2(object):

    def deleteString(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '

        def getPrefix(pattern, start):
            if False:
                while True:
                    i = 10
            prefix = [-1] * (len(pattern) - start)
            j = -1
            for i in xrange(1, len(pattern) - start):
                while j != -1 and pattern[start + j + 1] != pattern[start + i]:
                    j = prefix[j]
                if pattern[start + j + 1] == pattern[start + i]:
                    j += 1
                prefix[i] = j
            return prefix
        if all((x == s[0] for x in s)):
            return len(s)
        dp = [1] * len(s)
        for i in reversed(xrange(len(s) - 1)):
            prefix = getPrefix(s, i)
            for j in xrange(1, len(prefix), 2):
                if 2 * (prefix[j] + 1) == j + 1:
                    dp[i] = max(dp[i], dp[i + (prefix[j] + 1)] + 1)
        return dp[0]

class Solution3(object):

    def deleteString(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (MOD, P) = (10 ** 9 + 7, (113, 109))

        def hash(i, j):
            if False:
                print('Hello World!')
            return [(prefix[idx][j + 1] - prefix[idx][i] * power[idx][j - i + 1]) % MOD for idx in xrange(len(P))]
        if all((x == s[0] for x in s)):
            return len(s)
        power = [[1] for _ in xrange(len(P))]
        prefix = [[0] for _ in xrange(len(P))]
        for x in s:
            for (idx, p) in enumerate(P):
                power[idx].append(power[idx][-1] * p % MOD)
                prefix[idx].append((prefix[idx][-1] * p + (ord(x) - ord('a'))) % MOD)
        dp = [1] * len(s)
        for i in reversed(xrange(len(s) - 1)):
            for j in xrange(1, (len(s) - i) // 2 + 1):
                if hash(i, i + j - 1) == hash(i + j, i + 2 * j - 1):
                    dp[i] = max(dp[i], dp[i + j] + 1)
        return dp[0]
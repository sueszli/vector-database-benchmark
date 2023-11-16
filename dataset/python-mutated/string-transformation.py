class Solution(object):

    def numberOfWays(self, s, t, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type t: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def getPrefix(pattern):
            if False:
                for i in range(10):
                    print('nop')
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j + 1 > 0 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix

        def KMP(text, pattern):
            if False:
                i = 10
                return i + 15
            prefix = getPrefix(pattern)
            j = -1
            for i in xrange(len(text)):
                while j + 1 > 0 and pattern[j + 1] != text[i]:
                    j = prefix[j]
                if pattern[j + 1] == text[i]:
                    j += 1
                if j + 1 == len(pattern):
                    yield (i - j)
                    j = prefix[j]
        n = len(s)
        dp = [0] * 2
        dp[1] = (pow(n - 1, k, MOD) - (-1) ** k) * pow(n, MOD - 2, MOD) % MOD
        dp[0] = (dp[1] + (-1) ** k) % MOD
        return reduce(lambda a, b: (a + b) % MOD, (dp[int(i != 0)] for i in KMP(s + s[:-1], t)), 0)

class Solution2(object):

    def numberOfWays(self, s, t, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type t: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def matrix_mult(A, B):
            if False:
                while True:
                    i = 10
            ZB = zip(*B)
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]

        def matrix_expo(A, K):
            if False:
                print('Hello World!')
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result

        def getPrefix(pattern):
            if False:
                i = 10
                return i + 15
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j + 1 > 0 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix

        def KMP(text, pattern):
            if False:
                for i in range(10):
                    print('nop')
            prefix = getPrefix(pattern)
            j = -1
            for i in xrange(len(text)):
                while j + 1 > 0 and pattern[j + 1] != text[i]:
                    j = prefix[j]
                if pattern[j + 1] == text[i]:
                    j += 1
                if j + 1 == len(pattern):
                    yield (i - j)
                    j = prefix[j]
        n = len(s)
        T = [[0, 1], [n - 1, n - 1 - 1]]
        dp = [1, 0]
        dp = matrix_mult([dp], matrix_expo(T, k))[0]
        return reduce(lambda a, b: (a + b) % MOD, (dp[int(i != 0)] for i in KMP(s + s[:-1], t)), 0)

class Solution3(object):

    def numberOfWays(self, s, t, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def matrix_mult(A, B):
            if False:
                i = 10
                return i + 15
            ZB = zip(*B)
            return [[sum((a * b % MOD for (a, b) in itertools.izip(row, col))) % MOD for col in ZB] for row in A]

        def matrix_expo(A, K):
            if False:
                while True:
                    i = 10
            result = [[int(i == j) for j in xrange(len(A))] for i in xrange(len(A))]
            while K:
                if K % 2:
                    result = matrix_mult(result, A)
                A = matrix_mult(A, A)
                K /= 2
            return result

        def z_function(s):
            if False:
                while True:
                    i = 10
            z = [0] * len(s)
            (l, r) = (0, 0)
            for i in xrange(1, len(z)):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < len(z) and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    (l, r) = (i, i + z[i] - 1)
            return z
        n = len(s)
        T = [[0, 1], [n - 1, n - 1 - 1]]
        dp = [1, 0]
        dp = matrix_mult([dp], matrix_expo(T, k))[0]
        z = z_function(t + s + s[:-1])
        return reduce(lambda a, b: (a + b) % MOD, (dp[int(i != 0)] for i in xrange(n) if z[i + len(t)] >= len(t)), 0)
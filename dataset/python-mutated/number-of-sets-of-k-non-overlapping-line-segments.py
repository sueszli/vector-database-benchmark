MOD = 10 ** 9 + 7
MAX_N = 1000
fact = [0] * (2 * MAX_N - 1 + 1)
inv = [0] * (2 * MAX_N - 1 + 1)
inv_fact = [0] * (2 * MAX_N - 1 + 1)
fact[0] = inv_fact[0] = fact[1] = inv_fact[1] = inv[1] = 1
for i in xrange(2, len(fact)):
    fact[i] = fact[i - 1] * i % MOD
    inv[i] = inv[MOD % i] * (MOD - MOD // i) % MOD
    inv_fact[i] = inv_fact[i - 1] * inv[i] % MOD

class Solution(object):

    def numberOfSets(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '

        def nCr(n, k, mod):
            if False:
                print('Hello World!')
            return fact[n] * inv_fact[n - k] % mod * inv_fact[k] % mod
        return nCr(n + k - 1, 2 * k, MOD)

class Solution2(object):

    def numberOfSets(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def nCr(n, r):
            if False:
                print('Hello World!')
            if n - r < r:
                return nCr(n, n - r)
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c
        return nCr(n + k - 1, 2 * k) % MOD
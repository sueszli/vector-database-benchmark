import collections

class Solution(object):

    def countGoodSubsequences(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                for i in range(10):
                    print('nop')
            if not 0 <= k <= n:
                return 0
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD
        cnt = collections.Counter(s)
        return reduce(lambda total, k: (total + reduce(lambda total, x: total * (1 + nCr(x, k)) % MOD, cnt.itervalues(), 1) - 1) % MOD, xrange(1, max(cnt.itervalues()) + 1), 0)
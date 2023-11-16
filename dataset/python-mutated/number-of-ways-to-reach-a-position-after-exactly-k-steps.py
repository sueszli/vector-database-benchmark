class Solution(object):

    def numberOfWays(self, startPos, endPos, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type startPos: int\n        :type endPos: int\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                print('Hello World!')
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD
        r = k - abs(endPos - startPos)
        return nCr(k, r // 2) if r >= 0 and r % 2 == 0 else 0
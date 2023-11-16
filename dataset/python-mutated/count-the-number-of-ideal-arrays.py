import collections

class Solution(object):

    def idealArrays(self, n, maxValue):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type maxValue: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                return 10
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD

        def linear_sieve_of_eratosthenes(n):
            if False:
                print('Hello World!')
            primes = []
            spf = [-1] * (n + 1)
            for i in xrange(2, n + 1):
                if spf[i] == -1:
                    spf[i] = i
                    primes.append(i)
                for p in primes:
                    if i * p > n or p > spf[i]:
                        break
                    spf[i * p] = p
            return primes

        def prime_factors(x):
            if False:
                return 10
            factors = collections.Counter()
            for p in primes:
                if p * p > x:
                    break
                while x % p == 0:
                    factors[p] += 1
                    x //= p
            if x != 1:
                factors[x] += 1
            return factors
        primes = linear_sieve_of_eratosthenes(int(maxValue ** 0.5))
        result = 0
        for k in xrange(1, maxValue + 1):
            total = 1
            for c in prime_factors(k).itervalues():
                total = total * nCr(n + c - 1, c) % MOD
            result = (result + total) % MOD
        return result
import collections

class Solution2(object):

    def idealArrays(self, n, maxValue):
        if False:
            return 10
        '\n        :type n: int\n        :type maxValue: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                for i in range(10):
                    print('nop')
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD
        result = 0
        dp = collections.Counter(xrange(1, maxValue + 1))
        for i in xrange(n):
            new_dp = collections.Counter()
            total = 0
            for (x, c) in dp.iteritems():
                total = (total + c) % MOD
                for y in xrange(x + x, maxValue + 1, x):
                    new_dp[y] += c
            result = (result + total * nCr(n - 1, i)) % MOD
            dp = new_dp
        return result
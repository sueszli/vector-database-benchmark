import collections

class Solution(object):

    def waysToFillArray(self, queries):
        if False:
            return 10
        '\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                while True:
                    i = 10
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD

        def linear_sieve_of_eratosthenes(n):
            if False:
                while True:
                    i = 10
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
                i = 10
                return i + 15
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
        primes = linear_sieve_of_eratosthenes(int(max((k for (_, k) in queries)) ** 0.5))
        result = []
        for (n, k) in queries:
            total = 1
            for c in prime_factors(k).itervalues():
                total *= nCr(n + c - 1, c)
            result.append(total % MOD)
        return result
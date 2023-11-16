import collections

class Solution(object):

    def numberOfGoodSubsets(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def sieve_of_eratosthenes(n):
            if False:
                while True:
                    i = 10
            if n < 2:
                return []
            primes = [2]
            is_prime = [True] * ((n + 1) // 2)
            for i in xrange(1, len(is_prime)):
                if not is_prime[i]:
                    continue
                primes.append(2 * i + 1)
                for j in xrange(2 * i * (i + 1), len(is_prime), 2 * i + 1):
                    is_prime[j] = False
            return primes

        def to_mask(primes, x):
            if False:
                while True:
                    i = 10
            (mask, basis) = (0, 1)
            for p in primes:
                if x % p == 0:
                    mask |= basis
                basis <<= 1
            return mask
        MOD = 10 ** 9 + 7
        primes = sieve_of_eratosthenes(max(nums))
        dp = [0] * (1 << len(primes))
        dp[0] = 1
        cnts = collections.Counter(nums)
        for (x, cnt) in cnts.iteritems():
            if x == 1 or any((x % (p * p) == 0 for p in primes if p * p <= x)):
                continue
            mask = to_mask(primes, x)
            for i in xrange(len(dp) - 1):
                if i & mask:
                    continue
                dp[i | mask] = (dp[i | mask] + cnt * dp[i]) % MOD
        return pow(2, cnts[1], MOD) * (reduce(lambda total, x: (total + x) % MOD, dp, 0) - 1) % MOD
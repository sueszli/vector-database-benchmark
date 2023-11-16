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
MAX_N = 10 ** 3
PRIMES = linear_sieve_of_eratosthenes(int(MAX_N ** 0.5))

class Solution(object):

    def distinctPrimeFactors(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = set()
        for x in set(nums):
            for p in PRIMES:
                if p > x:
                    break
                if x % p:
                    continue
                result.add(p)
                while x % p == 0:
                    x //= p
            if x != 1:
                result.add(x)
        return len(result)
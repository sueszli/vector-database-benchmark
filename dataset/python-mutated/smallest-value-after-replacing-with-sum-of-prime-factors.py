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
MAX_N = 10 ** 5
PRIMES = linear_sieve_of_eratosthenes(int(MAX_N ** 0.5))

class Solution(object):

    def smallestValue(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        while True:
            (curr, new_n) = (n, 0)
            for p in PRIMES:
                if p ** 2 > curr:
                    break
                while curr % p == 0:
                    curr //= p
                    new_n += p
            if curr > 1:
                new_n += curr
            if new_n == n:
                break
            n = new_n
        return n
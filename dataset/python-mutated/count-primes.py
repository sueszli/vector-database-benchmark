class Solution(object):

    def countPrimes(self, n):
        if False:
            i = 10
            return i + 15
        if n <= 2:
            return 0
        is_prime = [True] * (n // 2)
        cnt = len(is_prime)
        for i in xrange(3, n, 2):
            if i * i >= n:
                break
            if not is_prime[i // 2]:
                continue
            for j in xrange(i * i, n, 2 * i):
                if not is_prime[j // 2]:
                    continue
                cnt -= 1
                is_prime[j // 2] = False
        return cnt

class Solution_TLE(object):

    def countPrimes(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '

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
        return len(linear_sieve_of_eratosthenes(n - 1))
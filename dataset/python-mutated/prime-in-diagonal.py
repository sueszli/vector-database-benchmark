def linear_sieve_of_eratosthenes(n):
    if False:
        i = 10
        return i + 15
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
MAX_N = 4 * 10 ** 6
PRIMES = linear_sieve_of_eratosthenes(MAX_N)
PRIMES_SET = set(PRIMES)

class Solution(object):

    def diagonalPrime(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[List[int]]\n        :rtype: int\n        '
        result = 0
        for i in xrange(len(nums)):
            if nums[i][i] in PRIMES_SET:
                result = max(result, nums[i][i])
            if nums[i][~i] in PRIMES_SET:
                result = max(result, nums[i][~i])
        return result
import bisect

def linear_sieve_of_eratosthenes(n):
    if False:
        for i in range(10):
            print('nop')
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
PRIMES = linear_sieve_of_eratosthenes(MAX_N - 1)

class Solution(object):

    def primeSubOperation(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        for i in xrange(len(nums)):
            j = bisect.bisect_left(PRIMES, nums[i] - nums[i - 1] if i - 1 >= 0 else nums[i])
            if j - 1 >= 0:
                nums[i] -= PRIMES[j - 1]
            if i - 1 >= 0 and nums[i - 1] >= nums[i]:
                return False
        return True
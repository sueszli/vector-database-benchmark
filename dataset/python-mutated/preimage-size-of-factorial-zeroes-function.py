class Solution(object):

    def preimageSizeFZF(self, K):
        if False:
            while True:
                i = 10
        '\n        :type K: int\n        :rtype: int\n        '

        def count_of_factorial_primes(n, p):
            if False:
                for i in range(10):
                    print('nop')
            cnt = 0
            while n > 0:
                cnt += n // p
                n //= p
            return cnt
        p = 5
        (left, right) = (0, p * K)
        while left <= right:
            mid = left + (right - left) // 2
            if count_of_factorial_primes(mid, p) >= K:
                right = mid - 1
            else:
                left = mid + 1
        return p if count_of_factorial_primes(left, p) == K else 0
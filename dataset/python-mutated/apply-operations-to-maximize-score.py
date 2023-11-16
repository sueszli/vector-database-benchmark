import heapq

class Solution(object):

    def maximumScore(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def linear_sieve_of_eratosthenes(n):
            if False:
                return 10
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
        lookup = {}

        def count_of_distinct_prime_factors(x):
            if False:
                return 10
            y = x
            if y not in lookup:
                cnt = 0
                for p in primes:
                    if p * p > x:
                        break
                    if x % p != 0:
                        continue
                    cnt += 1
                    while x % p == 0:
                        x //= p
                if x != 1:
                    cnt += 1
                lookup[y] = cnt
            return lookup[y]
        primes = linear_sieve_of_eratosthenes(int(max(nums) ** 0.5))
        scores = [count_of_distinct_prime_factors(x) for x in nums]
        left = [-1] * len(scores)
        stk = [-1]
        for i in xrange(len(scores)):
            while stk[-1] != -1 and scores[stk[-1]] < scores[i]:
                stk.pop()
            left[i] = stk[-1]
            stk.append(i)
        right = [-1] * len(scores)
        stk = [len(scores)]
        for i in reversed(xrange(len(scores))):
            while stk[-1] != len(scores) and scores[stk[-1]] <= scores[i]:
                stk.pop()
            right[i] = stk[-1]
            stk.append(i)
        result = 1
        max_heap = [(-x, i) for (i, x) in enumerate(nums)]
        heapq.heapify(max_heap)
        while max_heap:
            (_, i) = heapq.heappop(max_heap)
            c = min((i - left[i]) * (right[i] - i), k)
            result = result * pow(nums[i], c, MOD) % MOD
            k -= c
            if not k:
                break
        return result
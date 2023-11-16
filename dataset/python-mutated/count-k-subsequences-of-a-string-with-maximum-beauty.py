import collections
import random

class Solution(object):

    def countKSubsequencesWithMaxBeauty(self, s, k):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (fact, inv, inv_fact) = [[1] * 2 for _ in xrange(3)]

        def nCr(n, k):
            if False:
                for i in range(10):
                    print('nop')
            if not 0 <= k <= n:
                return 0
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                for i in range(10):
                    print('nop')

            def tri_partition(nums, left, right, target, compare):
                if False:
                    print('Hello World!')
                mid = left
                while mid <= right:
                    if nums[mid] == target:
                        mid += 1
                    elif compare(nums[mid], target):
                        (nums[left], nums[mid]) = (nums[mid], nums[left])
                        left += 1
                        mid += 1
                    else:
                        (nums[mid], nums[right]) = (nums[right], nums[mid])
                        right -= 1
                return (left, right)
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1
        cnt = collections.Counter(s)
        if len(cnt) < k:
            return 0
        freqs = cnt.values()
        nth_element(freqs, k - 1, lambda a, b: a > b)
        n = freqs.count(freqs[k - 1])
        r = sum((freqs[i] == freqs[k - 1] for i in xrange(k)))
        return reduce(lambda a, b: a * b % MOD, (freqs[i] for i in xrange(k)), 1) * nCr(n, r) % MOD
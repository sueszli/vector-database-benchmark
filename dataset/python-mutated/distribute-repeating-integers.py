import collections
import random

class Solution(object):

    def canDistribute(self, nums, quantity):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type quantity: List[int]\n        :rtype: bool\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                for i in range(10):
                    print('nop')

            def tri_partition(nums, left, right, target, compare):
                if False:
                    i = 10
                    return i + 15
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
        count = collections.Counter(nums)
        total = (1 << len(quantity)) - 1
        requirement = [0] * (total + 1)
        for mask in xrange(len(requirement)):
            base = 1
            for i in xrange(len(quantity)):
                if mask & base:
                    requirement[mask] += quantity[i]
                base <<= 1
        dp = [[0] * (total + 1) for _ in xrange(2)]
        dp[0][0] = 1
        i = 0
        cnts = count.values()
        if len(quantity) < len(cnts):
            nth_element(cnts, len(quantity) - 1, lambda a, b: a > b)
            cnts = cnts[:len(quantity)]
        for cnt in cnts:
            dp[(i + 1) % 2] = [0] * (total + 1)
            for mask in reversed(xrange(total + 1)):
                dp[(i + 1) % 2][mask] |= dp[i % 2][mask]
                submask = mask
                while submask > 0:
                    if requirement[submask] <= cnt and dp[i % 2][mask ^ submask]:
                        dp[(i + 1) % 2][mask] = 1
                    submask = submask - 1 & mask
            i += 1
        return dp[len(cnts) % 2][total]
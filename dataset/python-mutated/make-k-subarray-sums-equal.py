import random

class Solution(object):

    def makeSubKSumEqual(self, arr, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                while True:
                    i = 10
            while b:
                (a, b) = (b, a % b)
            return a

        def nth_element(nums, n, left=0, compare=lambda a, b: a < b):
            if False:
                return 10

            def tri_partition(nums, left, right, target, compare):
                if False:
                    while True:
                        i = 10
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
            right = len(nums) - 1
            while left <= right:
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1
        l = gcd(k, len(arr))
        result = 0
        for i in xrange(l):
            vals = [arr[j] for j in xrange(i, len(arr), l)]
            nth_element(vals, len(vals) // 2)
            result += sum((abs(v - vals[len(vals) // 2]) for v in vals))
        return result
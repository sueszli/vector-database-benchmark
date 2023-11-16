import random

class Solution(object):

    def putMarbles(self, weights, k):
        if False:
            print('Hello World!')
        '\n        :type weights: List[int]\n        :type k: int\n        :rtype: int\n        '

        def nth_element(nums, n, left=0, compare=lambda a, b: a < b):
            if False:
                while True:
                    i = 10

            def tri_partition(nums, left, right, target, compare):
                if False:
                    return 10
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
        for i in xrange(len(weights) - 1):
            weights[i] += weights[i + 1]
        weights.pop()
        result = 0
        nth_element(weights, k - 1 - 1, compare=lambda a, b: a > b)
        result += sum((weights[i] for i in xrange(k - 1)))
        nth_element(weights, k - 1 - 1)
        result -= sum((weights[i] for i in xrange(k - 1)))
        return result
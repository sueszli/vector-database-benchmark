import random

class Solution(object):

    def maxSubsequence(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                i = 10
                return i + 15

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
        partition = nums[:]
        nth_element(partition, k - 1, compare=lambda a, b: a > b)
        cnt = sum((partition[i] == partition[k - 1] for i in xrange(k)))
        result = []
        for x in nums:
            if x > partition[k - 1]:
                result.append(x)
            elif x == partition[k - 1] and cnt > 0:
                cnt -= 1
                result.append(x)
        return result
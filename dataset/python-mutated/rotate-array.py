class Solution(object):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """

    def rotate(self, nums, k):
        if False:
            print('Hello World!')

        def reverse(nums, start, end):
            if False:
                while True:
                    i = 10
            while start < end:
                (nums[start], nums[end - 1]) = (nums[end - 1], nums[start])
                start += 1
                end -= 1
        k %= len(nums)
        reverse(nums, 0, len(nums))
        reverse(nums, 0, k)
        reverse(nums, k, len(nums))
from fractions import gcd

class Solution2(object):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """

    def rotate(self, nums, k):
        if False:
            return 10

        def apply_cycle_permutation(k, offset, cycle_len, nums):
            if False:
                print('Hello World!')
            tmp = nums[offset]
            for i in xrange(1, cycle_len):
                (nums[(offset + i * k) % len(nums)], tmp) = (tmp, nums[(offset + i * k) % len(nums)])
            nums[offset] = tmp
        k %= len(nums)
        num_cycles = gcd(len(nums), k)
        cycle_len = len(nums) / num_cycles
        for i in xrange(num_cycles):
            apply_cycle_permutation(k, i, cycle_len, nums)

class Solution3(object):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """

    def rotate(self, nums, k):
        if False:
            print('Hello World!')
        count = 0
        start = 0
        while count < len(nums):
            curr = start
            prev = nums[curr]
            while True:
                idx = (curr + k) % len(nums)
                (nums[idx], prev) = (prev, nums[idx])
                curr = idx
                count += 1
                if start == curr:
                    break
            start += 1

class Solution4(object):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """

    def rotate(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        nums[:] = nums[len(nums) - k:] + nums[:len(nums) - k]

class Solution5(object):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """

    def rotate(self, nums, k):
        if False:
            i = 10
            return i + 15
        while k > 0:
            nums.insert(0, nums.pop())
            k -= 1
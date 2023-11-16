class Solution(object):

    def pivotIndex(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        total = sum(nums)
        left_sum = 0
        for (i, num) in enumerate(nums):
            if left_sum == total - left_sum - num:
                return i
            left_sum += num
        return -1
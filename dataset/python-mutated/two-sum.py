class Solution(object):

    def twoSum(self, nums, target):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        '
        lookup = {}
        for (i, num) in enumerate(nums):
            if target - num in lookup:
                return [lookup[target - num], i]
            lookup[num] = i

    def twoSum2(self, nums, target):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        '
        for i in nums:
            j = target - i
            tmp_nums_start_index = nums.index(i) + 1
            tmp_nums = nums[tmp_nums_start_index:]
            if j in tmp_nums:
                return [nums.index(i), tmp_nums_start_index + tmp_nums.index(j)]
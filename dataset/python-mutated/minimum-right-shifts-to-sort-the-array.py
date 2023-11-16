class Solution(object):

    def minimumRightShifts(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        i = next((i for i in xrange(len(nums)) if not nums[i] < nums[(i + 1) % len(nums)]), len(nums))
        j = next((j for j in xrange(i + 1, len(nums)) if not nums[j % len(nums)] < nums[(j + 1) % len(nums)]), len(nums))
        return len(nums) - (i + 1) if j == len(nums) else -1
class Solution(object):

    def checkArray(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '
        curr = 0
        for (i, x) in enumerate(nums):
            if x - curr < 0:
                return False
            nums[i] -= curr
            curr += nums[i]
            if i - (k - 1) >= 0:
                curr -= nums[i - (k - 1)]
        return curr == 0
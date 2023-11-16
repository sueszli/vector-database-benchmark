class Solution(object):

    def maxNumOfMarkedIndices(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        left = 0
        for right in xrange((len(nums) + 1) // 2, len(nums)):
            if nums[right] >= 2 * nums[left]:
                left += 1
        return left * 2

class Solution2(object):

    def maxNumOfMarkedIndices(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        left = 0
        for right in xrange(len(nums)):
            if nums[right] >= 2 * nums[left]:
                left += 1
        return min(left, len(nums) // 2) * 2
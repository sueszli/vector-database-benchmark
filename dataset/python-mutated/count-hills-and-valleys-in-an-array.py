class Solution(object):

    def countHillValley(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, inc) = (0, -1)
        for i in xrange(len(nums) - 1):
            if nums[i] < nums[i + 1]:
                result += int(inc == 0)
                inc = 1
            elif nums[i] > nums[i + 1]:
                result += int(inc == 1)
                inc = 0
        return result
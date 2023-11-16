class Solution(object):

    def optimalDivision(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: str\n        '
        if len(nums) == 1:
            return str(nums[0])
        if len(nums) == 2:
            return str(nums[0]) + '/' + str(nums[1])
        result = [str(nums[0]) + '/(' + str(nums[1])]
        for i in xrange(2, len(nums)):
            result += '/' + str(nums[i])
        result += ')'
        return ''.join(result)
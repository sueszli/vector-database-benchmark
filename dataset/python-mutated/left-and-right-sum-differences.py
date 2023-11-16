class Solution(object):

    def leftRigthDifference(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        total = sum(nums)
        result = []
        curr = 0
        for x in nums:
            curr += x
            result.append(abs(curr - x - (total - curr)))
        return result
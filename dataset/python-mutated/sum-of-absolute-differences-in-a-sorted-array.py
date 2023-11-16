class Solution(object):

    def getSumAbsoluteDifferences(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (prefix, suffix) = (0, sum(nums))
        result = []
        for (i, num) in enumerate(nums):
            suffix -= num
            result.append(i * num - prefix + (suffix - (len(nums) - 1 - i) * num))
            prefix += num
        return result
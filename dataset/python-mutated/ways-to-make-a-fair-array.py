class Solution(object):

    def waysToMakeFair(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        prefix = [0] * 2
        suffix = [sum((nums[i] for i in xrange(k, len(nums), 2))) for k in xrange(2)]
        result = 0
        for (i, num) in enumerate(nums):
            suffix[i % 2] -= num
            result += int(prefix[0] + suffix[1] == prefix[1] + suffix[0])
            prefix[i % 2] += num
        return result
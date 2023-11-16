class Solution(object):

    def findMissingRanges(self, nums, lower, upper):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type lower: int\n        :type upper: int\n        :rtype: List[str]\n        '

        def getRange(lower, upper):
            if False:
                while True:
                    i = 10
            if lower == upper:
                return '{}'.format(lower)
            else:
                return '{}->{}'.format(lower, upper)
        ranges = []
        pre = lower - 1
        for i in xrange(len(nums) + 1):
            if i == len(nums):
                cur = upper + 1
            else:
                cur = nums[i]
            if cur - pre >= 2:
                ranges.append(getRange(pre + 1, cur - 1))
            pre = cur
        return ranges
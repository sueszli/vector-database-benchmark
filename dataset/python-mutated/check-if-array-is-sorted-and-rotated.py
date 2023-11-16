class Solution(object):

    def check(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        count = 0
        for i in xrange(len(nums)):
            if nums[i] > nums[(i + 1) % len(nums)]:
                count += 1
                if count > 1:
                    return False
        return True
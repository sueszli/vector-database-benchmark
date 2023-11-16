class Solution(object):

    def maximizeGreatness(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return len(nums) - max(collections.Counter(nums).itervalues())

class Solution2(object):

    def maximizeGreatness(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        left = 0
        for right in xrange(len(nums)):
            if nums[right] > nums[left]:
                left += 1
        return left
class Solution(object):

    def buildArray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        for i in xrange(len(nums)):
            (prev, curr) = (i, nums[i])
            while curr >= 0 and curr != i:
                (nums[prev], nums[curr]) = (~nums[curr], ~nums[prev] if prev == i else nums[prev])
                (prev, curr) = (curr, ~nums[prev])
        for i in xrange(len(nums)):
            if nums[i] < 0:
                nums[i] = ~nums[i]
        return nums

class Solution2(object):

    def buildArray(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [nums[x] for x in nums]
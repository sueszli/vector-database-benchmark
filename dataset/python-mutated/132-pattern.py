class Solution(object):

    def find132pattern(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        ak = float('-inf')
        stk = []
        for i in reversed(xrange(len(nums))):
            if nums[i] < ak:
                return True
            while stk and stk[-1] < nums[i]:
                ak = stk.pop()
            stk.append(nums[i])
        return False

class Solution_TLE(object):

    def find132pattern(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        for k in xrange(len(nums)):
            valid = False
            for j in xrange(k):
                if nums[j] < nums[k]:
                    valid = True
                elif nums[j] > nums[k]:
                    if valid:
                        return True
        return False
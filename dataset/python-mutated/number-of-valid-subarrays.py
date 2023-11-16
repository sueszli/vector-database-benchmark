class Solution(object):

    def validSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        s = []
        for num in nums:
            while s and s[-1] > num:
                s.pop()
            s.append(num)
            result += len(s)
        return result
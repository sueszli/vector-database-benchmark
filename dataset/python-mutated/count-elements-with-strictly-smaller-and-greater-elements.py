class Solution(object):

    def countElements(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        mn = min(nums)
        mx = max(nums)
        return sum((mn < x < mx for x in nums))
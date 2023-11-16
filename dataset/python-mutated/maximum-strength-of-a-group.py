class Solution(object):

    def maxStrength(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        if all((x <= 0 for x in nums)) and sum((x < 0 for x in nums)) <= 1:
            return max(nums)
        result = reduce(lambda x, y: x * y, (x for x in nums if x))
        return result if result > 0 else result // max((x for x in nums if x < 0))
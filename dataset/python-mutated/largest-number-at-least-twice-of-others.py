class Solution(object):

    def dominantIndex(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        m = max(nums)
        if all((m >= 2 * x for x in nums if x != m)):
            return nums.index(m)
        return -1
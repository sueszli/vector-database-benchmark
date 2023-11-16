class Solution(object):

    def minimumSwaps(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        min_idx = min(xrange(len(nums)), key=nums.__getitem__)
        max_idx = max(reversed(xrange(len(nums))), key=nums.__getitem__)
        return len(nums) - 1 - max_idx + min_idx - int(max_idx < min_idx)
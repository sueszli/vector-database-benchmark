class Solution(object):

    def minPatches(self, nums, n):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type n: int\n        :rtype: int\n        '
        (patch, miss, i) = (0, 1, 0)
        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                miss += miss
                patch += 1
        return patch
class Solution(object):

    def smallestSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = [0] * len(nums)
        lookup = [-1] * max(max(nums).bit_length(), 1)
        for i in reversed(xrange(len(nums))):
            for bit in xrange(len(lookup)):
                if nums[i] & 1 << bit:
                    lookup[bit] = i
            result[i] = max(max(lookup) - i + 1, 1)
        return result
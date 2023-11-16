import collections

class Solution(object):

    def minLengthAfterRemovals(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        mx = max(collections.Counter(nums).itervalues())
        return mx - (len(nums) - mx) if mx > len(nums) - mx else len(nums) % 2
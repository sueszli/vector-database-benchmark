class Solution(object):

    def findKDistantIndices(self, nums, key, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type key: int\n        :type k: int\n        :rtype: List[int]\n        '
        result = []
        prev = -1
        for (i, x) in enumerate(nums):
            if x != key:
                continue
            for j in xrange(max(i - k, prev + 1), min(i + k + 1, len(nums))):
                result.append(j)
            prev = min(i + k, len(nums) - 1)
        return result
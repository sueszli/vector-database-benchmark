class Solution(object):

    def minOperations(self, nums, queries):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type queries: List[int]\n        :rtype: List[int]\n        '
        nums.sort()
        prefix = [0] * (len(nums) + 1)
        for i in xrange(len(nums)):
            prefix[i + 1] = prefix[i] + nums[i]
        result = [0] * len(queries)
        for (i, q) in enumerate(queries):
            j = bisect.bisect_left(nums, q)
            result[i] = q * j - prefix[j] + (prefix[-1] - prefix[j] - q * (len(nums) - j))
        return result
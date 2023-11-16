class Solution(object):

    def partitionArray(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        nums.sort()
        (result, prev) = (1, 0)
        for i in xrange(len(nums)):
            if nums[i] - nums[prev] <= k:
                continue
            prev = i
            result += 1
        return result
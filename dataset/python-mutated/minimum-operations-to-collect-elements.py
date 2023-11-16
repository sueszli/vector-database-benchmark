class Solution(object):

    def minOperations(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        lookup = [False] * k
        for i in reversed(xrange(len(nums))):
            if nums[i] > len(lookup) or lookup[nums[i] - 1]:
                continue
            lookup[nums[i] - 1] = True
            k -= 1
            if not k:
                break
        return len(nums) - i
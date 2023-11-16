import bisect

class Solution(object):

    def minimumMountainRemovals(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        left_lis_len = [0] * len(nums)
        lis = []
        for i in xrange(len(nums) - 1):
            j = bisect.bisect_left(lis, nums[i])
            if j == len(lis):
                lis.append(nums[i])
            else:
                lis[j] = nums[i]
            left_lis_len[i] = j
        max_len = 0
        lis = []
        for i in reversed(xrange(1, len(nums))):
            j = bisect.bisect_left(lis, nums[i])
            if j == len(lis):
                lis.append(nums[i])
            else:
                lis[j] = nums[i]
            if i < len(nums) - 1:
                max_len = max(max_len, left_lis_len[i] + j)
        return len(nums) - (1 + max_len)
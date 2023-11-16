class Solution(object):

    def countWays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        cnt = [0] * (len(nums) + 1)
        for x in nums:
            cnt[x] += 1
        result = prefix = 0
        for i in xrange(len(nums) + 1):
            if prefix == i and cnt[i] == 0:
                result += 1
            prefix += cnt[i]
        return result

class Solution2(object):

    def countWays(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        return sum(((i == 0 or nums[i - 1] < i) and (i == len(nums) or nums[i] > i) for i in xrange(len(nums) + 1)))
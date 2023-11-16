class Solution(object):

    def countQuadruplets(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = [0] * len(nums)
        result = 0
        for l in xrange(len(nums)):
            cnt = 0
            for j in xrange(l):
                if nums[j] < nums[l]:
                    cnt += 1
                    result += dp[j]
                elif nums[j] > nums[l]:
                    dp[j] += cnt
        return result

class Solution2(object):

    def countQuadruplets(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        right = [[0] * (len(nums) + 1) for _ in xrange(len(nums))]
        for j in xrange(len(nums)):
            for i in reversed(xrange(j + 1, len(nums))):
                right[j][i] = right[j][i + 1] + int(nums[i] > nums[j])
        result = 0
        for k in xrange(len(nums)):
            left = 0
            for j in xrange(k):
                if nums[k] < nums[j]:
                    result += left * right[j][k + 1]
                left += int(nums[k] > nums[j])
        return result

class Solution3(object):

    def countQuadruplets(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        left = [[0] * (len(nums) + 1) for _ in xrange(len(nums))]
        for j in xrange(len(nums)):
            for i in xrange(j):
                left[j][i + 1] = left[j][i] + int(nums[i] < nums[j])
        right = [[0] * (len(nums) + 1) for _ in xrange(len(nums))]
        for j in xrange(len(nums)):
            for i in reversed(xrange(j + 1, len(nums))):
                right[j][i] = right[j][i + 1] + int(nums[i] > nums[j])
        result = 0
        for k in xrange(len(nums)):
            for j in xrange(k):
                if nums[k] < nums[j]:
                    result += left[k][j] * right[j][k + 1]
        return result
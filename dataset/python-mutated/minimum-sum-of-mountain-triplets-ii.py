class Solution(object):

    def minimumSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        INF = float('inf')
        right = [INF] * len(nums)
        curr = INF
        for i in reversed(xrange(len(nums))):
            right[i] = curr
            curr = min(curr, nums[i])
        result = curr = INF
        for i in xrange(len(nums)):
            if curr < nums[i] > right[i]:
                result = min(result, curr + nums[i] + right[i])
            curr = min(curr, nums[i])
        return result if result != INF else -1

class Solution2(object):

    def minimumSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        INF = float('inf')
        left = [INF] * len(nums)
        curr = INF
        for i in xrange(len(nums)):
            left[i] = curr
            curr = min(curr, nums[i])
        right = [INF] * len(nums)
        curr = INF
        for i in reversed(xrange(len(nums))):
            right[i] = curr
            curr = min(curr, nums[i])
        result = INF
        for i in xrange(len(nums)):
            if left[i] < nums[i] > right[i]:
                result = min(result, left[i] + nums[i] + right[i])
        return result if result != INF else -1
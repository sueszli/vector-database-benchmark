class Solution(object):

    def kthSmallestSubarraySum(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(nums, k, x):
            if False:
                return 10
            cnt = curr = left = 0
            for right in xrange(len(nums)):
                curr += nums[right]
                while curr > x:
                    curr -= nums[left]
                    left += 1
                cnt += right - left + 1
            return cnt >= k
        (left, right) = (min(nums), sum(nums))
        while left <= right:
            mid = left + (right - left) // 2
            if check(nums, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
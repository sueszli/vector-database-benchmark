class Solution(object):

    def splitArray(self, nums, m):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type m: int\n        :rtype: int\n        '

        def check(nums, m, s):
            if False:
                i = 10
                return i + 15
            (cnt, curr_sum) = (1, 0)
            for num in nums:
                curr_sum += num
                if curr_sum > s:
                    curr_sum = num
                    cnt += 1
            return cnt <= m
        (left, right) = (max(nums), sum(nums))
        while left <= right:
            mid = left + (right - left) // 2
            if check(nums, m, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
class Solution(object):

    def countFairPairs(self, nums, lower, upper):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type lower: int\n        :type upper: int\n        :rtype: int\n        '

        def count(x):
            if False:
                return 10
            cnt = 0
            (left, right) = (0, len(nums) - 1)
            while left < right:
                if nums[left] + nums[right] <= x:
                    cnt += right - left
                    left += 1
                else:
                    right -= 1
            return cnt
        nums.sort()
        return count(upper) - count(lower - 1)
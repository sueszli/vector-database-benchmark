class Solution(object):

    def smallestDistancePair(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def possible(guess, nums, k):
            if False:
                print('Hello World!')
            (count, left) = (0, 0)
            for (right, num) in enumerate(nums):
                while num - nums[left] > guess:
                    left += 1
                count += right - left
            return count >= k
        nums.sort()
        (left, right) = (0, nums[-1] - nums[0] + 1)
        while left < right:
            mid = left + (right - left) / 2
            if possible(mid, nums, k):
                right = mid
            else:
                left = mid + 1
        return left
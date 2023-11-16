class Solution(object):

    def minCapability(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(x):
            if False:
                return 10
            cnt = i = 0
            while i < len(nums):
                if nums[i] <= x:
                    cnt += 1
                    i += 2
                else:
                    i += 1
            return cnt >= k
        sorted_nums = sorted(set(nums))
        (left, right) = (0, len(sorted_nums) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if check(sorted_nums[mid]):
                right = mid - 1
            else:
                left = mid + 1
        return sorted_nums[left]

class Solution2(object):

    def minCapability(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(x):
            if False:
                print('Hello World!')
            cnt = i = 0
            while i < len(nums):
                if nums[i] <= x:
                    cnt += 1
                    i += 2
                else:
                    i += 1
            return cnt >= k
        (left, right) = (min(nums), max(nums))
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
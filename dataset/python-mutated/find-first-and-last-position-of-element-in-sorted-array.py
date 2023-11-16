class Solution(object):

    def searchRange(self, nums, target):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        '

        def binarySearch(n, check):
            if False:
                return 10
            (left, right) = (0, n - 1)
            while left <= right:
                mid = left + (right - left) // 2
                if check(mid):
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def binarySearch2(n, check):
            if False:
                i = 10
                return i + 15
            (left, right) = (0, n)
            while left < right:
                mid = left + (right - left) // 2
                if check(mid):
                    right = mid
                else:
                    left = mid + 1
            return left

        def binarySearch3(n, check):
            if False:
                return 10
            (left, right) = (-1, n - 1)
            while left < right:
                mid = right - (right - left) // 2
                if check(mid):
                    right = mid - 1
                else:
                    left = mid
            return left + 1

        def binarySearch4(n, check):
            if False:
                while True:
                    i = 10
            (left, right) = (-1, n)
            while right - left >= 2:
                mid = left + (right - left) // 2
                if check(mid):
                    right = mid
                else:
                    left = mid
            return right
        left = binarySearch(len(nums), lambda i: nums[i] >= target)
        if left == len(nums) or nums[left] != target:
            return [-1, -1]
        right = binarySearch(len(nums), lambda i: nums[i] > target)
        return [left, right - 1]
class Solution(object):

    def minimumSize(self, nums, maxOperations):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type maxOperations: int\n        :rtype: int\n        '

        def check(nums, maxOperations, x):
            if False:
                print('Hello World!')
            return sum(((num + x - 1) // x - 1 for num in nums)) <= maxOperations
        (left, right) = (1, max(nums))
        while left <= right:
            mid = left + (right - left) // 2
            if check(nums, maxOperations, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
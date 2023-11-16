class Solution(object):

    def smallestDivisor(self, nums, threshold):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type threshold: int\n        :rtype: int\n        '

        def check(A, d, threshold):
            if False:
                for i in range(10):
                    print('nop')
            return sum(((i - 1) // d + 1 for i in nums)) <= threshold
        (left, right) = (1, max(nums))
        while left <= right:
            mid = left + (right - left) // 2
            if check(nums, mid, threshold):
                right = mid - 1
            else:
                left = mid + 1
        return left
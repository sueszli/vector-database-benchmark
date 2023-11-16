class Solution(object):

    def sortTransformedArray(self, nums, a, b, c):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: List[int]\n        '
        f = lambda x, a, b, c: a * x * x + b * x + c
        result = []
        if not nums:
            return result
        (left, right) = (0, len(nums) - 1)
        d = -1 if a > 0 else 1
        while left <= right:
            if d * f(nums[left], a, b, c) < d * f(nums[right], a, b, c):
                result.append(f(nums[left], a, b, c))
                left += 1
            else:
                result.append(f(nums[right], a, b, c))
                right -= 1
        return result[::d]
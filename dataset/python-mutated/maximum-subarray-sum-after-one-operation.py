class Solution(object):

    def maxSumAfterOperation(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        prev_with_square = prev_without_square = 0
        result = 0
        for num in nums:
            without_square = max(num, num + prev_without_square)
            with_square = max(num * num, num * num + prev_without_square, num + prev_with_square)
            result = max(result, with_square)
            (prev_with_square, prev_without_square) = (with_square, without_square)
        return result
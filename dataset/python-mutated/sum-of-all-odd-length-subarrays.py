class Solution(object):

    def sumOddLengthSubarrays(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                print('Hello World!')
            return (a + (b - 1)) // b
        return sum((x * ceil_divide((i - 0 + 1) * (len(arr) - 1 - i + 1), 2) for (i, x) in enumerate(arr)))
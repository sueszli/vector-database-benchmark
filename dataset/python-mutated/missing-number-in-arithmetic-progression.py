class Solution(object):

    def missingNumber(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '

        def check(arr, d, x):
            if False:
                while True:
                    i = 10
            return arr[x] != arr[0] + d * x
        d = (arr[-1] - arr[0]) // len(arr)
        (left, right) = (0, len(arr) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if check(arr, d, mid):
                right = mid - 1
            else:
                left = mid + 1
        return arr[0] + d * left

class Solution2(object):

    def missingNumber(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        return (min(arr) + max(arr)) * (len(arr) + 1) // 2 - sum(arr)
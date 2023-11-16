class Solution(object):

    def findBestValue(self, arr, target):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :type target: int\n        :rtype: int\n        '
        arr.sort(reverse=True)
        max_arr = arr[0]
        while arr and arr[-1] * len(arr) <= target:
            target -= arr.pop()
        return max_arr if not arr else (2 * target + len(arr) - 1) // (2 * len(arr))

class Solution2(object):

    def findBestValue(self, arr, target):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :type target: int\n        :rtype: int\n        '
        arr.sort(reverse=True)
        max_arr = arr[0]
        while arr and arr[-1] * len(arr) <= target:
            target -= arr.pop()
        if not arr:
            return max_arr
        x = (target - 1) // len(arr)
        return x if target - x * len(arr) <= (x + 1) * len(arr) - target else x + 1

class Solution3(object):

    def findBestValue(self, arr, target):
        if False:
            print('Hello World!')
        '\n        :type arr: List[int]\n        :type target: int\n        :rtype: int\n        '

        def total(arr, v):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            for x in arr:
                result += min(v, x)
            return result

        def check(arr, v, target):
            if False:
                for i in range(10):
                    print('nop')
            return total(arr, v) >= target
        (left, right) = (1, max(arr))
        while left <= right:
            mid = left + (right - left) // 2
            if check(arr, mid, target):
                right = mid - 1
            else:
                left = mid + 1
        return left - 1 if target - total(arr, left - 1) <= total(arr, left) - target else left
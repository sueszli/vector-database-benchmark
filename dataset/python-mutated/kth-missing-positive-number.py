class Solution(object):

    def findKthPositive(self, arr, k):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(arr, k, x):
            if False:
                return 10
            return arr[x] - (x + 1) < k
        (left, right) = (0, len(arr) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(arr, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right + 1 + k
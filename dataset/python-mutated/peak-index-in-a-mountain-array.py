class Solution(object):

    def peakIndexInMountainArray(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        (left, right) = (0, len(arr) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] > arr[mid + 1]:
                right = mid - 1
            else:
                left = mid + 1
        return left
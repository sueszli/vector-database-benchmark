import bisect

class Solution(object):

    def findSpecialInteger(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        for x in [arr[len(arr) // 4], arr[len(arr) // 2], arr[len(arr) * 3 // 4]]:
            if (bisect.bisect_right(arr, x) - bisect.bisect_left(arr, x)) * 4 > len(arr):
                return x
        return -1
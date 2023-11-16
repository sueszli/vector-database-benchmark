import bisect

class Solution(object):

    def findTheDistanceValue(self, arr1, arr2, d):
        if False:
            return 10
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :type d: int\n        :rtype: int\n        '
        arr2.sort()
        (result, i, j) = (0, 0, 0)
        for x in arr1:
            j = bisect.bisect_left(arr2, x)
            left = arr2[j - 1] if j - 1 >= 0 else float('-inf')
            right = arr2[j] if j < len(arr2) else float('inf')
            result += left + d < x < right - d
        return result

class Solution2(object):

    def findTheDistanceValue(self, arr1, arr2, d):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :type d: int\n        :rtype: int\n        '
        (arr1.sort(), arr2.sort())
        (result, i, j) = (0, 0, 0)
        while i < len(arr1) and j < len(arr2):
            if arr1[i] - arr2[j] > d:
                j += 1
                continue
            result += arr2[j] - arr1[i] > d
            i += 1
        return result + len(arr1) - i
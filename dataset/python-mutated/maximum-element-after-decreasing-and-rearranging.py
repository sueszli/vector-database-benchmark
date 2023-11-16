class Solution(object):

    def maximumElementAfterDecrementingAndRearranging(self, arr):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        arr.sort()
        result = 1
        for i in xrange(1, len(arr)):
            result = min(result + 1, arr[i])
        return result
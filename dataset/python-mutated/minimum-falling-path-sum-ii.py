import heapq

class Solution(object):

    def minFallingPathSum(self, arr):
        if False:
            print('Hello World!')
        '\n        :type arr: List[List[int]]\n        :rtype: int\n        '
        for i in xrange(1, len(arr)):
            smallest_two = heapq.nsmallest(2, arr[i - 1])
            for j in xrange(len(arr[0])):
                arr[i][j] += smallest_two[1] if arr[i - 1][j] == smallest_two[0] else smallest_two[0]
        return min(arr[-1])
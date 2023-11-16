class Solution(object):

    def getWinner(self, arr, k):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '
        result = arr[0]
        count = 0
        for i in xrange(1, len(arr)):
            if arr[i] > result:
                result = arr[i]
                count = 0
            count += 1
            if count == k:
                break
        return result
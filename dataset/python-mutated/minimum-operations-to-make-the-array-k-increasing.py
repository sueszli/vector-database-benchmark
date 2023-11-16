import bisect

class Solution(object):

    def kIncreasing(self, arr, k):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '

        def longest_non_decreasing_subsequence(arr):
            if False:
                while True:
                    i = 10
            result = []
            for x in arr:
                right = bisect.bisect_right(result, x)
                if right == len(result):
                    result.append(x)
                else:
                    result[right] = x
            return len(result)
        return len(arr) - sum((longest_non_decreasing_subsequence((arr[j] for j in xrange(i, len(arr), k))) for i in xrange(k)))
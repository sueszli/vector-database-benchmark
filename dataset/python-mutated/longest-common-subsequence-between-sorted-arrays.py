class Solution(object):

    def longestCommomSubsequence(self, arrays):
        if False:
            while True:
                i = 10
        '\n        :type arrays: List[List[int]]\n        :rtype: List[int]\n        '
        result = min(arrays, key=lambda x: len(x))
        for arr in arrays:
            new_result = []
            (i, j) = (0, 0)
            while i != len(result) and j != len(arr):
                if result[i] < arr[j]:
                    i += 1
                elif result[i] > arr[j]:
                    j += 1
                else:
                    new_result.append(result[i])
                    i += 1
                    j += 1
            result = new_result
        return result
import collections

class Solution2(object):

    def longestCommomSubsequence(self, arrays):
        if False:
            print('Hello World!')
        '\n        :type arrays: List[List[int]]\n        :rtype: List[int]\n        '
        return [num for (num, cnt) in collections.Counter((x for arr in arrays for x in arr)).iteritems() if cnt == len(arrays)]
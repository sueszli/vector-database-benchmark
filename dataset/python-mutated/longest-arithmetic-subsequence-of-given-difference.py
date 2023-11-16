import collections

class Solution(object):

    def longestSubsequence(self, arr, difference):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :type difference: int\n        :rtype: int\n        '
        result = 1
        lookup = collections.defaultdict(int)
        for i in xrange(len(arr)):
            lookup[arr[i]] = lookup[arr[i] - difference] + 1
            result = max(result, lookup[arr[i]])
        return result
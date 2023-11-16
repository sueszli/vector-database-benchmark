import collections

class Solution(object):

    def findLucky(self, arr):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        count = collections.Counter(arr)
        result = -1
        for (k, v) in count.iteritems():
            if k == v:
                result = max(result, k)
        return result
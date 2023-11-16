import collections

class Solution(object):

    def findLeastNumOfUniqueInts(self, arr, k):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '
        count = collections.Counter(arr)
        (result, count_count) = (len(count), collections.Counter(count.itervalues()))
        for c in xrange(1, len(arr) + 1):
            if k < c * count_count[c]:
                result -= k // c
                break
            k -= c * count_count[c]
            result -= count_count[c]
        return result
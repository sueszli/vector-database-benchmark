import collections

class Solution(object):

    def canArrange(self, arr, k):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: bool\n        '
        count = collections.Counter((i % k for i in arr))
        return (0 not in count or not count[0] % 2) and all((k - i in count and count[i] == count[k - i] for i in xrange(1, k) if i in count))
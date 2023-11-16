import collections

class Solution(object):

    def reorderedPowerOf2(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: bool\n        '
        count = collections.Counter(str(N))
        return any((count == collections.Counter(str(1 << i)) for i in xrange(31)))
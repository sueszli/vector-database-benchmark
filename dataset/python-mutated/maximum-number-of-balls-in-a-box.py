import collections
import itertools

class Solution(object):

    def countBalls(self, lowLimit, highLimit):
        if False:
            print('Hello World!')
        '\n        :type lowLimit: int\n        :type highLimit: int\n        :rtype: int\n        '
        count = collections.Counter()
        for i in xrange(lowLimit, highLimit + 1):
            count[sum(itertools.imap(int, str(i)))] += 1
        return max(count.itervalues())
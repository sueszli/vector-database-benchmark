import collections

class Solution(object):

    def checkStrings(self, s1, s2):
        if False:
            while True:
                i = 10
        '\n        :type s1: str\n        :type s2: str\n        :rtype: bool\n        '
        return all((collections.Counter((s1[j] for j in xrange(i, len(s1), 2))) == collections.Counter((s2[j] for j in xrange(i, len(s2), 2))) for i in xrange(2)))
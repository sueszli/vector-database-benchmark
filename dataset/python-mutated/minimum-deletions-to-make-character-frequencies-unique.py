import collections
import string

class Solution(object):

    def minDeletions(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        count = collections.Counter(s)
        result = 0
        lookup = set()
        for c in string.ascii_lowercase:
            for i in reversed(xrange(1, count[c] + 1)):
                if i not in lookup:
                    lookup.add(i)
                    break
                result += 1
        return result
import collections

class Solution(object):

    def numberOfSubstrings(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = 0
        cnt = collections.Counter()
        for c in s:
            cnt[c] += 1
            result += cnt[c]
        return result
import collections

class Solution(object):

    def numberOfSubstrings(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        return sum((v * (v + 1) // 2 for v in collections.Counter(s).itervalues()))
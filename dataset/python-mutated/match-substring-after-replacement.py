import collections

class Solution(object):

    def matchReplacement(self, s, sub, mappings):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type sub: str\n        :type mappings: List[List[str]]\n        :rtype: bool\n        '

        def transform(x):
            if False:
                print('Hello World!')
            return ord(x) - ord('0') if x.isdigit() else ord(x) - ord('a') + 10 if x.islower() else ord(x) - ord('A') + 36

        def check(i):
            if False:
                i = 10
                return i + 15
            return all((sub[j] == s[i + j] or lookup[sub[j]][s[i + j]] for j in xrange(len(sub))))
        lookup = [[0] * 62 for _ in xrange(62)]
        for (a, b) in mappings:
            lookup[transform(a)][transform(b)] = 1
        s = map(transform, s)
        sub = map(transform, sub)
        return any((check(i) for i in xrange(len(s) - len(sub) + 1)))
import collections

class Solution2(object):

    def matchReplacement(self, s, sub, mappings):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type sub: str\n        :type mappings: List[List[str]]\n        :rtype: bool\n        '

        def check(i):
            if False:
                print('Hello World!')
            return all((sub[j] == s[i + j] or (sub[j], s[i + j]) in lookup for j in xrange(len(sub))))
        lookup = set()
        for (a, b) in mappings:
            lookup.add((a, b))
        return any((check(i) for i in xrange(len(s) - len(sub) + 1)))
from itertools import izip

class Solution(object):

    def isIsomorphic(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        if len(s) != len(t):
            return False
        (s2t, t2s) = ({}, {})
        for (p, w) in izip(s, t):
            if w not in s2t and p not in t2s:
                s2t[w] = p
                t2s[p] = w
            elif w not in s2t or s2t[w] != p:
                return False
        return True

class Solution2(object):

    def isIsomorphic(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        if len(s) != len(t):
            return False
        return self.halfIsom(s, t) and self.halfIsom(t, s)

    def halfIsom(self, s, t):
        if False:
            print('Hello World!')
        lookup = {}
        for i in xrange(len(s)):
            if s[i] not in lookup:
                lookup[s[i]] = t[i]
            elif lookup[s[i]] != t[i]:
                return False
        return True
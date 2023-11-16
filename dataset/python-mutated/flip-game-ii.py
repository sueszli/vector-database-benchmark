import itertools
import re

class Solution(object):

    def canWin(self, s):
        if False:
            for i in range(10):
                print('nop')
        (g, g_final) = ([0], 0)
        for p in itertools.imap(len, re.split('-+', s)):
            while len(g) <= p:
                g += (min(set(xrange(p)) - {x ^ y for (x, y) in itertools.izip(g[:len(g) / 2], g[-2:-len(g) / 2 - 2:-1])}),)
            g_final ^= g[p]
        return g_final > 0

class Solution2(object):

    def canWin(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: bool\n        '
        lookup = {}

        def canWinHelper(consecutives):
            if False:
                i = 10
                return i + 15
            consecutives = tuple(sorted((c for c in consecutives if c >= 2)))
            if consecutives not in lookup:
                lookup[consecutives] = any((not canWinHelper(consecutives[:i] + (j, c - 2 - j) + consecutives[i + 1:]) for (i, c) in enumerate(consecutives) for j in xrange(c - 1)))
            return lookup[consecutives]
        return canWinHelper(map(len, re.findall('\\+\\++', s)))

class Solution3(object):

    def canWin(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: bool\n        '
        (i, n) = (0, len(s) - 1)
        is_win = False
        while not is_win and i < n:
            if s[i] == '+':
                while not is_win and i < n and (s[i + 1] == '+'):
                    is_win = not self.canWin(s[:i] + '--' + s[i + 2:])
                    i += 1
            i += 1
        return is_win
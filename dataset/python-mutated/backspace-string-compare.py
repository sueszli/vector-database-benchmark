import itertools

class Solution(object):

    def backspaceCompare(self, S, T):
        if False:
            return 10
        '\n        :type S: str\n        :type T: str\n        :rtype: bool\n        '

        def findNextChar(S):
            if False:
                while True:
                    i = 10
            skip = 0
            for i in reversed(xrange(len(S))):
                if S[i] == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield S[i]
        return all((x == y for (x, y) in itertools.izip_longest(findNextChar(S), findNextChar(T))))
class Solution(object):

    def generatePossibleNextMoves(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: List[str]\n        '
        res = []
        (i, n) = (0, len(s) - 1)
        while i < n:
            if s[i] == '+':
                while i < n and s[i + 1] == '+':
                    res.append(s[:i] + '--' + s[i + 2:])
                    i += 1
            i += 1
        return res

class Solution2(object):

    def generatePossibleNextMoves(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n      :type s: str\n      :rtype: List[str]\n      '
        return [s[:i] + '--' + s[i + 2:] for i in xrange(len(s) - 1) if s[i:i + 2] == '++']
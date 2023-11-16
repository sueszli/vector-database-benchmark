class Solution(object):

    def minimumMoves(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        result = i = 0
        while i < len(s):
            if s[i] == 'X':
                result += 1
                i += 3
            else:
                i += 1
        return result
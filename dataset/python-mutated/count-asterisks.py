class Solution(object):

    def countAsterisks(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        result = cnt = 0
        for c in s:
            if c == '|':
                cnt = (cnt + 1) % 2
                continue
            if c == '*' and cnt == 0:
                result += 1
        return result
class Solution(object):

    def checkValidString(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: bool\n        '
        (lower, upper) = (0, 0)
        for c in s:
            lower += 1 if c == '(' else -1
            upper -= 1 if c == ')' else -1
            if upper < 0:
                break
            lower = max(lower, 0)
        return lower == 0
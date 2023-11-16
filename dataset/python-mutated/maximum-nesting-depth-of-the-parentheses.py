class Solution(object):

    def maxDepth(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        result = curr = 0
        for c in s:
            if c == '(':
                curr += 1
                result = max(result, curr)
            elif c == ')':
                curr -= 1
        return result
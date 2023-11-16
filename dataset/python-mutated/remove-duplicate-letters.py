from collections import Counter

class Solution(object):

    def removeDuplicateLetters(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '
        remaining = Counter(s)
        (in_stack, stk) = (set(), [])
        for c in s:
            if c not in in_stack:
                while stk and stk[-1] > c and remaining[stk[-1]]:
                    in_stack.remove(stk.pop())
                stk += c
                in_stack.add(c)
            remaining[c] -= 1
        return ''.join(stk)
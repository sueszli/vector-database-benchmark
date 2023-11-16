import collections

class Solution(object):

    def smallestSubsequence(self, text):
        if False:
            while True:
                i = 10
        '\n        :type text: str\n        :rtype: str\n        '
        count = collections.Counter(text)
        (lookup, stk) = (set(), [])
        for c in text:
            if c not in lookup:
                while stk and stk[-1] > c and count[stk[-1]]:
                    lookup.remove(stk.pop())
                stk += c
                lookup.add(c)
            count[c] -= 1
        return ''.join(stk)
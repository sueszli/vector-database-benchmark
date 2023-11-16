class Solution(object):

    def smallestSubsequence(self, s, k, letter, repetition):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :type letter: str\n        :type repetition: int\n        :rtype: str\n        '
        stk = []
        suffix = [0] * (len(s) + 1)
        for i in reversed(xrange(len(suffix) - 1)):
            suffix[i] = suffix[i + 1] + (s[i] == letter)
        for (i, c) in enumerate(s):
            while stk and stk[-1] > c and (len(stk) + (len(s) - i) > k) and (stk[-1] != letter or repetition + 1 <= suffix[i]):
                repetition += stk.pop() == letter
            if len(stk) < min(k - (repetition - (c == letter)), k):
                repetition -= c == letter
                stk.append(c)
        return ''.join(stk)
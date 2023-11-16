import collections

class Solution(object):

    def robotWithString(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: str\n        '
        cnt = collections.Counter(s)
        (result, stk) = ([], [])
        mn = 'a'
        for c in s:
            stk.append(c)
            cnt[c] -= 1
            while mn < 'z' and cnt[mn] == 0:
                mn = chr(ord(mn) + 1)
            while stk and stk[-1] <= mn:
                result.append(stk.pop())
        return ''.join(result)
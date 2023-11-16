class Solution(object):

    def minimumScore(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :rtype: int\n        '
        right = [-1] * len(s)
        j = len(t) - 1
        for i in reversed(xrange(len(s))):
            if j >= 0 and t[j] == s[i]:
                j -= 1
            right[i] = j
        result = j + 1
        left = 0
        for i in xrange(len(s)):
            result = max(min(result, right[i] - left + 1), 0)
            if left < len(t) and t[left] == s[i]:
                left += 1
        result = min(result, len(t) - left)
        return result
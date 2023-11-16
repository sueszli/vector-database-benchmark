class Solution(object):

    def longestPalindrome(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: str\n        '

        def preProcess(s):
            if False:
                i = 10
                return i + 15
            if not s:
                return ['^', '$']
            T = ['^']
            for c in s:
                T += ['#', c]
            T += ['#', '$']
            return T
        T = preProcess(s)
        P = [0] * len(T)
        (center, right) = (0, 0)
        for i in xrange(1, len(T) - 1):
            i_mirror = 2 * center - i
            if right > i:
                P[i] = min(right - i, P[i_mirror])
            else:
                P[i] = 0
            while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
                P[i] += 1
            if i + P[i] > right:
                (center, right) = (i, i + P[i])
        max_i = 0
        for i in xrange(1, len(T) - 1):
            if P[i] > P[max_i]:
                max_i = i
        start = (max_i - 1 - P[max_i]) // 2
        return s[start:start + P[max_i]]

class Solution2(object):

    def longestPalindrome(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: str\n        '

        def expand(s, left, right):
            if False:
                for i in range(10):
                    print('nop')
            while left >= 0 and right < len(s) and (s[left] == s[right]):
                left -= 1
                right += 1
            return right - left + 1 - 2
        (left, right) = (-1, -2)
        for i in xrange(len(s)):
            l = max(expand(s, i, i), expand(s, i, i + 1))
            if l > right - left + 1:
                right = i + l // 2
                left = right - l + 1
        return s[left:right + 1] if left >= 0 else ''
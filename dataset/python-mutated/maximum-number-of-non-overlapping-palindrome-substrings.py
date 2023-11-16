class Solution(object):

    def maxPalindromes(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        result = prev = 0
        for mid in xrange(2 * len(s) - 1):
            (left, right) = (mid // 2, mid // 2 + mid % 2)
            while left >= prev and right < len(s) and (s[left] == s[right]):
                if right - left + 1 >= k:
                    prev = right + 1
                    result += 1
                    break
                (left, right) = (left - 1, right + 1)
        return result
class Solution(object):

    def maximumSubsequenceCount(self, text, pattern):
        if False:
            i = 10
            return i + 15
        '\n        :type text: str\n        :type pattern: str\n        :rtype: int\n        '
        result = cnt1 = cnt2 = 0
        for c in text:
            if c == pattern[1]:
                result += cnt1
                cnt2 += 1
            if c == pattern[0]:
                cnt1 += 1
        return result + max(cnt1, cnt2)
class Solution(object):

    def takeCharacters(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        cnt = [0] * 3
        for c in s:
            cnt[ord(c) - ord('a')] += 1
        if min(cnt) < k:
            return -1
        result = left = 0
        for right in xrange(len(s)):
            cnt[ord(s[right]) - ord('a')] -= 1
            while cnt[ord(s[right]) - ord('a')] < k:
                cnt[ord(s[left]) - ord('a')] += 1
                left += 1
            result = max(result, right - left + 1)
        return len(s) - result
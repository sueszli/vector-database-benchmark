class Solution(object):

    def maxVowels(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        VOWELS = set('aeiou')
        result = curr = 0
        for (i, c) in enumerate(s):
            curr += c in VOWELS
            if i >= k:
                curr -= s[i - k] in VOWELS
            result = max(result, curr)
        return result
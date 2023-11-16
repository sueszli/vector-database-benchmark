import collections

class Solution(object):

    def maxFreq(self, s, maxLetters, minSize, maxSize):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type maxLetters: int\n        :type minSize: int\n        :type maxSize: int\n        :rtype: int\n        '
        (M, p) = (10 ** 9 + 7, 113)
        (power, rolling_hash) = (pow(p, minSize - 1, M), 0)
        left = 0
        (lookup, count) = (collections.defaultdict(int), collections.defaultdict(int))
        for right in xrange(len(s)):
            count[s[right]] += 1
            if right - left + 1 > minSize:
                count[s[left]] -= 1
                rolling_hash = (rolling_hash - ord(s[left]) * power) % M
                if count[s[left]] == 0:
                    count.pop(s[left])
                left += 1
            rolling_hash = (rolling_hash * p + ord(s[right])) % M
            if right - left + 1 == minSize and len(count) <= maxLetters:
                lookup[rolling_hash] += 1
        return max(lookup.values() or [0])

class Solution2(object):

    def maxFreq(self, s, maxLetters, minSize, maxSize):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type maxLetters: int\n        :type minSize: int\n        :type maxSize: int\n        :rtype: int\n        '
        lookup = {}
        for right in xrange(minSize - 1, len(s)):
            word = s[right - minSize + 1:right + 1]
            if word in lookup:
                lookup[word] += 1
            elif len(collections.Counter(word)) <= maxLetters:
                lookup[word] = 1
        return max(lookup.values() or [0])
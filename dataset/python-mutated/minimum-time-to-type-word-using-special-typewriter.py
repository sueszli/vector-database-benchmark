class Solution(object):

    def minTimeToType(self, word):
        if False:
            i = 10
            return i + 15
        '\n        :type word: str\n        :rtype: int\n        '
        return min((ord(word[0]) - ord('a')) % 26, (ord('a') - ord(word[0])) % 26) + 1 + sum((min((ord(word[i]) - ord(word[i - 1])) % 26, (ord(word[i - 1]) - ord(word[i])) % 26) + 1 for i in xrange(1, len(word))))
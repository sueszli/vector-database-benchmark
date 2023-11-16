class Solution(object):

    def countDistinct(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        count = 0
        trie = {}
        for i in xrange(len(s)):
            curr = trie
            for j in xrange(i, len(s)):
                if s[j] not in curr:
                    count += 1
                    curr[s[j]] = {}
                curr = curr[s[j]]
        return count
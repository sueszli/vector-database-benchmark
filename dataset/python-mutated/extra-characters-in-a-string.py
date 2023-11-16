import collections

class Solution(object):

    def minExtraChar(self, s, dictionary):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type dictionary: List[str]\n        :rtype: int\n        '
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for word in dictionary:
            reduce(dict.__getitem__, word, trie).setdefault('_end')
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0
        for i in xrange(len(s)):
            dp[i + 1] = min(dp[i + 1], dp[i] + 1)
            curr = trie
            for j in xrange(i, len(s)):
                if s[j] not in curr:
                    break
                curr = curr[s[j]]
                if '_end' in curr:
                    dp[j + 1] = min(dp[j + 1], dp[i])
        return dp[-1]
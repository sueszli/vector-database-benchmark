class Solution(object):

    def getWordsInLongestSubsequence(self, n, words, groups):
        if False:
            return 10
        '\n        :type n: int\n        :type words: List[str]\n        :type groups: List[int]\n        :rtype: List[str]\n        '
        return [words[i] for i in xrange(n) if i == 0 or groups[i - 1] != groups[i]]
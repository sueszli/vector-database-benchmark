import collections

class Solution(object):

    def maximumNumberOfStringPairs(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: int\n        '
        result = 0
        cnt = collections.Counter()
        for w in words:
            result += cnt[w[::-1]]
            cnt[w] += 1
        return result
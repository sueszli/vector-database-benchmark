import bisect

class Solution(object):

    def numSmallerByFrequency(self, queries, words):
        if False:
            while True:
                i = 10
        '\n        :type queries: List[str]\n        :type words: List[str]\n        :rtype: List[int]\n        '
        words_freq = sorted((word.count(min(word)) for word in words))
        return [len(words) - bisect.bisect_right(words_freq, query.count(min(query))) for query in queries]
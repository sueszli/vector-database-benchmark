import collections

class WordDistance(object):

    def __init__(self, words):
        if False:
            while True:
                i = 10
        self.wordIndex = collections.defaultdict(list)
        for i in xrange(len(words)):
            self.wordIndex[words[i]].append(i)

    def shortest(self, word1, word2):
        if False:
            i = 10
            return i + 15
        indexes1 = self.wordIndex[word1]
        indexes2 = self.wordIndex[word2]
        (i, j, dist) = (0, 0, float('inf'))
        while i < len(indexes1) and j < len(indexes2):
            dist = min(dist, abs(indexes1[i] - indexes2[j]))
            if indexes1[i] < indexes2[j]:
                i += 1
            else:
                j += 1
        return dist
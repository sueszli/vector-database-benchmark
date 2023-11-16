class Solution(object):

    def isAlienSorted(self, words, order):
        if False:
            return 10
        '\n        :type words: List[str]\n        :type order: str\n        :rtype: bool\n        '
        lookup = {c: i for (i, c) in enumerate(order)}
        for i in xrange(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]
            for k in xrange(min(len(word1), len(word2))):
                if word1[k] != word2[k]:
                    if lookup[word1[k]] > lookup[word2[k]]:
                        return False
                    break
            else:
                if len(word1) > len(word2):
                    return False
        return True
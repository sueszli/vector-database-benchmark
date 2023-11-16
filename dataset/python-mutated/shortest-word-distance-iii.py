class Solution(object):

    def shortestWordDistance(self, words, word1, word2):
        if False:
            for i in range(10):
                print('nop')
        dist = float('inf')
        is_same = word1 == word2
        (i, index1, index2) = (0, None, None)
        while i < len(words):
            if words[i] == word1:
                if is_same and index1 is not None:
                    dist = min(dist, abs(index1 - i))
                index1 = i
            elif words[i] == word2:
                index2 = i
            if index1 is not None and index2 is not None:
                dist = min(dist, abs(index1 - index2))
            i += 1
        return dist
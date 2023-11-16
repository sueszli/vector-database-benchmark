class Solution(object):

    def wordCount(self, startWords, targetWords):
        if False:
            i = 10
            return i + 15
        '\n        :type startWords: List[str]\n        :type targetWords: List[str]\n        :rtype: int\n        '

        def bitmask(w):
            if False:
                return 10
            return reduce(lambda x, y: x | y, (1 << ord(c) - ord('a') for (i, c) in enumerate(w)))
        lookup = set((bitmask(w) for w in startWords))
        result = 0
        for w in targetWords:
            mask = bitmask(w)
            result += any((mask ^ 1 << ord(c) - ord('a') in lookup for c in w))
        return result
import itertools

class Solution(object):

    def expressiveWords(self, S, words):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :type words: List[str]\n        :rtype: int\n        '

        def RLE(S):
            if False:
                print('Hello World!')
            return itertools.izip(*[(k, len(list(grp))) for (k, grp) in itertools.groupby(S)])
        (R, count) = RLE(S)
        result = 0
        for word in words:
            (R2, count2) = RLE(word)
            if R2 != R:
                continue
            result += all((c1 >= max(c2, 3) or c1 == c2 for (c1, c2) in itertools.izip(count, count2)))
        return result
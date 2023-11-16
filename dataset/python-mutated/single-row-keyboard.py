class Solution(object):

    def calculateTime(self, keyboard, word):
        if False:
            print('Hello World!')
        '\n        :type keyboard: str\n        :type word: str\n        :rtype: int\n        '
        lookup = {c: i for (i, c) in enumerate(keyboard)}
        (result, prev) = (0, 0)
        for c in word:
            result += abs(lookup[c] - prev)
            prev = lookup[c]
        return result
import itertools

class Solution(object):

    def isAcronym(self, words, s):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :type s: str\n        :rtype: bool\n        '
        return len(words) == len(s) and all((w[0] == c for (w, c) in itertools.izip(words, s)))
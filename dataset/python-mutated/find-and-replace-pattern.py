import itertools

class Solution(object):

    def findAndReplacePattern(self, words, pattern):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :type pattern: str\n        :rtype: List[str]\n        '

        def match(word):
            if False:
                i = 10
                return i + 15
            lookup = {}
            for (x, y) in itertools.izip(pattern, word):
                if lookup.setdefault(x, y) != y:
                    return False
            return len(set(lookup.values())) == len(lookup.values())
        return filter(match, words)
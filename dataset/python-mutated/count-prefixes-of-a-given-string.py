import itertools

class Solution(object):

    def countPrefixes(self, words, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :type s: str\n        :rtype: int\n        '
        return sum(itertools.imap(s.startswith, words))
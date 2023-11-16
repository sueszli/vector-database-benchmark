import itertools

class Solution(object):

    def sortSentence(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: str\n        '
        words = s.split()
        for i in xrange(len(words)):
            while int(words[i][-1]) - 1 != i:
                (words[int(words[i][-1]) - 1], words[i]) = (words[i], words[int(words[i][-1]) - 1])
        return ' '.join(itertools.imap(lambda x: x[:-1], words))
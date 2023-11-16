class Solution(object):

    def addMinimum(self, word):
        if False:
            print('Hello World!')
        '\n        :type word: str\n        :rtype: int\n        '
        return 3 * sum((i - 1 < 0 or word[i - 1] >= word[i] for i in xrange(len(word)))) - len(word)
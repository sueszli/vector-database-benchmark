class Solution(object):

    def isCircularSentence(self, sentence):
        if False:
            return 10
        '\n        :type sentence: str\n        :rtype: bool\n        '
        return sentence[0] == sentence[-1] and all((sentence[i - 1] == sentence[i + 1] for i in xrange(len(sentence)) if sentence[i] == ' '))
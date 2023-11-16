class Solution(object):

    def countValidWords(self, sentence):
        if False:
            print('Hello World!')
        '\n        :type sentence: str\n        :rtype: int\n        '
        result = token = hyphen = 0
        for i in xrange(len(sentence) + 1):
            if i == len(sentence) or sentence[i] == ' ':
                if token == 1:
                    result += 1
                token = hyphen = 0
                continue
            if sentence[i].isdigit() or (sentence[i] in '!.,' and (not (i == len(sentence) - 1 or sentence[i + 1] == ' '))) or (sentence[i] == '-' and (not (hyphen == 0 and 0 < i < len(sentence) - 1 and sentence[i - 1].isalpha() and sentence[i + 1].isalpha()))):
                token = -1
                continue
            if token == 0:
                token = 1
            if sentence[i] == '-':
                hyphen = 1
        return result
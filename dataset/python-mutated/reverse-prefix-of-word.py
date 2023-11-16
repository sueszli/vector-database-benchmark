class Solution(object):

    def reversePrefix(self, word, ch):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word: str\n        :type ch: str\n        :rtype: str\n        '
        i = word.find(ch)
        return word[:i + 1][::-1] + word[i + 1:]
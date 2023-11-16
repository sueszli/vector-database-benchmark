class Solution(object):

    def isSumEqual(self, firstWord, secondWord, targetWord):
        if False:
            i = 10
            return i + 15
        '\n        :type firstWord: str\n        :type secondWord: str\n        :type targetWord: str\n        :rtype: bool\n        '

        def stoi(s):
            if False:
                return 10
            result = 0
            for c in s:
                result = result * 10 + ord(c) - ord('a')
            return result
        return stoi(firstWord) + stoi(secondWord) == stoi(targetWord)
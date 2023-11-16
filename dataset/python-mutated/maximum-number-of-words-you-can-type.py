class Solution(object):

    def canBeTypedWords(self, text, brokenLetters):
        if False:
            i = 10
            return i + 15
        '\n        :type text: str\n        :type brokenLetters: str\n        :rtype: int\n        '
        lookup = set(brokenLetters)
        (result, broken) = (0, False)
        for c in text:
            if c == ' ':
                result += int(broken == False)
                broken = False
            elif c in lookup:
                broken = True
        return result + int(broken == False)
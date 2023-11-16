class Solution(object):

    def countVowels(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word: str\n        :rtype: int\n        '
        VOWELS = set('aeiou')
        return sum(((i - 0 + 1) * (len(word) - 1 - i + 1) for (i, c) in enumerate(word) if c in VOWELS))
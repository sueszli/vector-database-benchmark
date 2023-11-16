class Solution(object):

    def toGoatLatin(self, S):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :rtype: str\n        '

        def convert(S):
            if False:
                i = 10
                return i + 15
            vowel = set('aeiouAEIOU')
            for (i, word) in enumerate(S.split(), 1):
                if word[0] not in vowel:
                    word = word[1:] + word[:1]
                yield (word + 'ma' + 'a' * i)
        return ' '.join(convert(S))
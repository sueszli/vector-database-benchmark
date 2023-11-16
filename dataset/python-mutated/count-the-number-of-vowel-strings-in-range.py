class Solution(object):

    def vowelStrings(self, words, left, right):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :type left: int\n        :type right: int\n        :rtype: int\n        '
        VOWELS = {'a', 'e', 'i', 'o', 'u'}
        return sum((words[i][0] in VOWELS and words[i][-1] in VOWELS for i in xrange(left, right + 1)))
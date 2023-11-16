class Solution(object):

    def vowelStrings(self, words, queries):
        if False:
            return 10
        '\n        :type words: List[str]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        VOWELS = {'a', 'e', 'i', 'o', 'u'}
        prefix = [0] * (len(words) + 1)
        for (i, w) in enumerate(words):
            prefix[i + 1] = prefix[i] + int(w[0] in VOWELS and w[-1] in VOWELS)
        return [prefix[r + 1] - prefix[l] for (l, r) in queries]
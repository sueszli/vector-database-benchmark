class Solution(object):

    def findLongestWord(self, s, d):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type d: List[str]\n        :rtype: str\n        '
        d.sort(key=lambda x: (-len(x), x))
        for word in d:
            i = 0
            for c in s:
                if i < len(word) and word[i] == c:
                    i += 1
            if i == len(word):
                return word
        return ''
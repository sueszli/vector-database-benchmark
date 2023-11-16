class Solution(object):

    def divisibilityArray(self, word, m):
        if False:
            i = 10
            return i + 15
        '\n        :type word: str\n        :type m: int\n        :rtype: List[int]\n        '
        result = []
        curr = 0
        for c in word:
            curr = (curr * 10 + (ord(c) - ord('0'))) % m
            result.append(int(curr == 0))
        return result
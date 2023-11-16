class Solution(object):

    def isPrefixString(self, s, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type words: List[str]\n        :rtype: bool\n        '
        i = j = 0
        for c in s:
            if i == len(words) or words[i][j] != c:
                return False
            j += 1
            if j == len(words[i]):
                i += 1
                j = 0
        return j == 0

class Solution2(object):

    def isPrefixString(self, s, words):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type words: List[str]\n        :rtype: bool\n        '
        i = 0
        for word in words:
            for c in word:
                if i == len(s) or s[i] != c:
                    return False
                i += 1
            if i == len(s):
                return True
        return False
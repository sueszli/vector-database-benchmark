class Solution(object):

    def countConsistentStrings(self, allowed, words):
        if False:
            print('Hello World!')
        '\n        :type allowed: str\n        :type words: List[str]\n        :rtype: int\n        '
        lookup = [False] * 26
        for c in allowed:
            lookup[ord(c) - ord('a')] = True
        result = len(words)
        for word in words:
            for c in word:
                if not lookup[ord(c) - ord('a')]:
                    result -= 1
                    break
        return result
class Solution(object):

    def canConstruct(self, ransomNote, magazine):
        if False:
            while True:
                i = 10
        '\n        :type ransomNote: str\n        :type magazine: str\n        :rtype: bool\n        '
        counts = [0] * 26
        letters = 0
        for c in ransomNote:
            if counts[ord(c) - ord('a')] == 0:
                letters += 1
            counts[ord(c) - ord('a')] += 1
        for c in magazine:
            counts[ord(c) - ord('a')] -= 1
            if counts[ord(c) - ord('a')] == 0:
                letters -= 1
                if letters == 0:
                    break
        return letters == 0
import collections

class Solution2(object):

    def canConstruct(self, ransomNote, magazine):
        if False:
            print('Hello World!')
        '\n        :type ransomNote: str\n        :type magazine: str\n        :rtype: bool\n        '
        return not collections.Counter(ransomNote) - collections.Counter(magazine)
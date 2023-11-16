import collections

class Solution(object):

    def isAnagram(self, s, t):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        if len(s) != len(t):
            return False
        count = collections.defaultdict(int)
        for c in s:
            count[c] += 1
        for c in t:
            count[c] -= 1
            if count[c] < 0:
                return False
        return True

class Solution2(object):

    def isAnagram(self, s, t):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        return collections.Counter(s) == collections.Counter(t)

class Solution3(object):

    def isAnagram(self, s, t):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        return sorted(s) == sorted(t)
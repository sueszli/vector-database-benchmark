import itertools

class Solution(object):

    def areAlmostEqual(self, s1, s2):
        if False:
            print('Hello World!')
        '\n        :type s1: str\n        :type s2: str\n        :rtype: bool\n        '
        diff = []
        for (a, b) in itertools.izip(s1, s2):
            if a == b:
                continue
            if len(diff) == 2:
                return False
            diff.append([a, b] if not diff else [b, a])
        return not diff or (len(diff) == 2 and diff[0] == diff[1])
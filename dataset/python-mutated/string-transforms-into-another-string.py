import itertools

class Solution(object):

    def canConvert(self, str1, str2):
        if False:
            return 10
        '\n        :type str1: str\n        :type str2: str\n        :rtype: bool\n        '
        if str1 == str2:
            return True
        lookup = {}
        for (i, j) in itertools.izip(str1, str2):
            if lookup.setdefault(i, j) != j:
                return False
        return len(set(str2)) < 26
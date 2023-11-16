class Solution(object):

    def findLUSlength(self, a, b):
        if False:
            i = 10
            return i + 15
        '\n        :type a: str\n        :type b: str\n        :rtype: int\n        '
        if a == b:
            return -1
        return max(len(a), len(b))
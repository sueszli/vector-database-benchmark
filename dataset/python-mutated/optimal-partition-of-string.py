class Solution(object):

    def partitionString(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        (result, left) = (1, 0)
        lookup = {}
        for (i, x) in enumerate(s):
            if x in lookup and lookup[x] >= left:
                left = i
                result += 1
            lookup[x] = i
        return result
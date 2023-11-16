class Solution(object):

    def repeatedCharacter(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: str\n        '
        lookup = set()
        for c in s:
            if c in lookup:
                break
            lookup.add(c)
        return c
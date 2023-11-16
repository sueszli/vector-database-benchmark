class Solution(object):

    def addSpaces(self, s, spaces):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type spaces: List[int]\n        :rtype: str\n        '
        prev = len(s)
        s = list(s)
        s.extend([None] * len(spaces))
        for i in reversed(xrange(len(spaces))):
            for j in reversed(xrange(spaces[i], prev)):
                s[j + 1 + i] = s[j]
            s[spaces[i] + i] = ' '
            prev = spaces[i]
        return ''.join(s)
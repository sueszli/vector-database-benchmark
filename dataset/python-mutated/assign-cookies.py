class Solution(object):

    def findContentChildren(self, g, s):
        if False:
            i = 10
            return i + 15
        '\n        :type g: List[int]\n        :type s: List[int]\n        :rtype: int\n        '
        g.sort()
        s.sort()
        (result, i) = (0, 0)
        for j in xrange(len(s)):
            if i == len(g):
                break
            if s[j] >= g[i]:
                result += 1
                i += 1
        return result
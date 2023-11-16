class Solution(object):

    def checkDistances(self, s, distance):
        if False:
            return 10
        '\n        :type s: str\n        :type distance: List[int]\n        :rtype: bool\n        '
        for i in xrange(len(s)):
            if i + distance[ord(s[i]) - ord('a')] + 1 >= len(s) or s[i + distance[ord(s[i]) - ord('a')] + 1] != s[i]:
                return False
            distance[ord(s[i]) - ord('a')] = -1
        return True
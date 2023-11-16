class Solution(object):

    def cellsInRange(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: List[str]\n        '
        return [chr(x) + chr(y) for x in xrange(ord(s[0]), ord(s[3]) + 1) for y in xrange(ord(s[1]), ord(s[4]) + 1)]
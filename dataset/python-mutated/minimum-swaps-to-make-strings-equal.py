class Solution(object):

    def minimumSwap(self, s1, s2):
        if False:
            while True:
                i = 10
        '\n        :type s1: str\n        :type s2: str\n        :rtype: int\n        '
        (x1, y1) = (0, 0)
        for i in xrange(len(s1)):
            if s1[i] == s2[i]:
                continue
            x1 += int(s1[i] == 'x')
            y1 += int(s1[i] == 'y')
        if x1 % 2 != y1 % 2:
            return -1
        return x1 // 2 + y1 // 2 + (x1 % 2 + y1 % 2)
class Solution(object):

    def maximumOddBinaryNumber(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: str\n        '
        a = list(s)
        left = 0
        for i in xrange(len(a)):
            if a[i] != '1':
                continue
            (a[i], a[left]) = (a[left], a[i])
            left += 1
        if a[-1] != '1':
            (a[-1], a[left - 1]) = (a[left - 1], a[-1])
        return ''.join(a)

class Solution2(object):

    def maximumOddBinaryNumber(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '
        n = s.count('1')
        return ''.join(['1'] * (n - 1) + ['0'] * (len(s) - n) + ['1'])
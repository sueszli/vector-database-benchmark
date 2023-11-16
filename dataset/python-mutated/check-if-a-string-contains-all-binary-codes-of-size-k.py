class Solution(object):

    def hasAllCodes(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: bool\n        '
        return 2 ** k <= len(s) and len({s[i:i + k] for i in xrange(len(s) - k + 1)}) == 2 ** k

class Solution2(object):

    def hasAllCodes(self, s, k):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type k: int\n        :rtype: bool\n        '
        lookup = set()
        base = 2 ** k
        if base > len(s):
            return False
        num = 0
        for i in xrange(len(s)):
            num = (num << 1) + (s[i] == '1')
            if i >= k - 1:
                lookup.add(num)
                num -= (s[i - k + 1] == '1') * (base // 2)
        return len(lookup) == base
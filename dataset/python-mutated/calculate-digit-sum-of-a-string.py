class Solution(object):

    def digitSum(self, s, k):
        if False:
            return 10
        '\n        :type s: str\n        :type k: int\n        :rtype: str\n        '
        while len(s) > k:
            s = ''.join(map(str, (sum(map(int, s[i:i + k])) for i in xrange(0, len(s), k))))
        return s
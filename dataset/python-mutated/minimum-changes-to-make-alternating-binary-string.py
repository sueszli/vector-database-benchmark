class Solution(object):

    def minOperations(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        cnt = sum((int(c) == i % 2 for (i, c) in enumerate(s)))
        return min(cnt, len(s) - cnt)
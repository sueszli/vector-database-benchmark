class Solution(object):

    def checkZeroOnes(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: bool\n        '
        max_cnt = [0] * 2
        cnt = 0
        for i in xrange(len(s) + 1):
            if i == len(s) or (i >= 1 and s[i] != s[i - 1]):
                max_cnt[int(s[i - 1])] = max(max_cnt[int(s[i - 1])], cnt)
                cnt = 0
            cnt += 1
        return max_cnt[0] < max_cnt[1]
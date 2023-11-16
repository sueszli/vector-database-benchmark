class Solution(object):

    def minimumDeletions(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        result = b_cnt = 0
        for c in s:
            if c == 'b':
                b_cnt += 1
            elif b_cnt:
                b_cnt -= 1
                result += 1
        return result
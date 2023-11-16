class Solution(object):

    def countSubstrings(self, s, t):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type t: str\n        :rtype: int\n        '

        def count(i, j):
            if False:
                for i in range(10):
                    print('nop')
            result = left_cnt = right_cnt = 0
            for k in xrange(min(len(s) - i, len(t) - j)):
                right_cnt += 1
                if s[i + k] != t[j + k]:
                    (left_cnt, right_cnt) = (right_cnt, 0)
                result += left_cnt
            return result
        return sum((count(i, 0) for i in xrange(len(s)))) + sum((count(0, j) for j in xrange(1, len(t))))
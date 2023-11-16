class Solution(object):

    def isMatch(self, s, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type p: str\n        :rtype: bool\n        '
        count = 0
        (p_ptr, s_ptr, last_s_ptr, last_p_ptr) = (0, 0, -1, -1)
        while s_ptr < len(s):
            if p_ptr < len(p) and (s[s_ptr] == p[p_ptr] or p[p_ptr] == '?'):
                s_ptr += 1
                p_ptr += 1
            elif p_ptr < len(p) and p[p_ptr] == '*':
                p_ptr += 1
                last_s_ptr = s_ptr
                last_p_ptr = p_ptr
            elif last_p_ptr != -1:
                last_s_ptr += 1
                s_ptr = last_s_ptr
                p_ptr = last_p_ptr
            else:
                assert count <= (len(p) + 1) * (len(s) + 1)
                return False
            count += 1
        while p_ptr < len(p) and p[p_ptr] == '*':
            p_ptr += 1
            count += 1
        assert count <= (len(p) + 1) * (len(s) + 1)
        return p_ptr == len(p)

class Solution2(object):

    def isMatch(self, s, p):
        if False:
            while True:
                i = 10
        k = 2
        result = [[False for j in xrange(len(p) + 1)] for i in xrange(k)]
        result[0][0] = True
        for i in xrange(1, len(p) + 1):
            if p[i - 1] == '*':
                result[0][i] = result[0][i - 1]
        for i in xrange(1, len(s) + 1):
            result[i % k][0] = False
            for j in xrange(1, len(p) + 1):
                if p[j - 1] != '*':
                    result[i % k][j] = result[(i - 1) % k][j - 1] and (s[i - 1] == p[j - 1] or p[j - 1] == '?')
                else:
                    result[i % k][j] = result[i % k][j - 1] or result[(i - 1) % k][j]
        return result[len(s) % k][len(p)]

class Solution3(object):

    def isMatch(self, s, p):
        if False:
            i = 10
            return i + 15
        result = [[False for j in xrange(len(p) + 1)] for i in xrange(len(s) + 1)]
        result[0][0] = True
        for i in xrange(1, len(p) + 1):
            if p[i - 1] == '*':
                result[0][i] = result[0][i - 1]
        for i in xrange(1, len(s) + 1):
            result[i][0] = False
            for j in xrange(1, len(p) + 1):
                if p[j - 1] != '*':
                    result[i][j] = result[i - 1][j - 1] and (s[i - 1] == p[j - 1] or p[j - 1] == '?')
                else:
                    result[i][j] = result[i][j - 1] or result[i - 1][j]
        return result[len(s)][len(p)]

class Solution4(object):

    def isMatch(self, s, p):
        if False:
            for i in range(10):
                print('nop')
        if not p or not s:
            return not s and (not p)
        if p[0] != '*':
            if p[0] == s[0] or p[0] == '?':
                return self.isMatch(s[1:], p[1:])
            else:
                return False
        else:
            while len(s) > 0:
                if self.isMatch(s, p[1:]):
                    return True
                s = s[1:]
            return self.isMatch(s, p[1:])
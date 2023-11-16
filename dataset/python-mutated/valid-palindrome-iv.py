class Solution(object):

    def makePalindrome(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: bool\n        '
        return sum((s[i] != s[~i] for i in xrange(len(s) // 2))) <= 2

class Solution2(object):

    def makePalindrome(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: bool\n        '
        cnt = 0
        (left, right) = (0, len(s) - 1)
        while left < right:
            if s[left] != s[right]:
                cnt += 1
                if cnt > 2:
                    return False
            left += 1
            right -= 1
        return True
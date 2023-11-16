class Solution(object):

    def appealSum(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        result = curr = 0
        lookup = [-1] * 26
        for (i, c) in enumerate(s):
            result += (i - lookup[ord(c) - ord('a')]) * (len(s) - i)
            lookup[ord(c) - ord('a')] = i
        return result

class Solution2(object):

    def appealSum(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        result = cnt = 0
        lookup = [-1] * 26
        for (i, c) in enumerate(s):
            cnt += i - lookup[ord(c) - ord('a')]
            lookup[ord(c) - ord('a')] = i
            result += cnt
        return result
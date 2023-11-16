class Solution(object):

    def maximumScore(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: int\n        '
        return min((a + b + c) // 2, a + b + c - max(a, b, c))
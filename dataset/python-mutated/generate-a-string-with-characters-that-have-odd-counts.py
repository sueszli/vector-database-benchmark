class Solution(object):

    def generateTheString(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: str\n        '
        result = ['a'] * (n - 1)
        result.append('a' if n % 2 else 'b')
        return ''.join(result)
class Solution(object):

    def convertToTitle(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: str\n        '
        result = []
        while n:
            result += chr((n - 1) % 26 + ord('A'))
            n = (n - 1) // 26
        result.reverse()
        return ''.join(result)
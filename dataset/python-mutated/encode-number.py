class Solution(object):

    def encode(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: str\n        '
        result = []
        while num:
            result.append('0' if num % 2 else '1')
            num = (num - 1) // 2
        return ''.join(reversed(result))
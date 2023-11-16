class Solution(object):

    def countDigits(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: int\n        '
        result = 0
        curr = num
        while curr:
            result += int(num % (curr % 10) == 0)
            curr //= 10
        return result
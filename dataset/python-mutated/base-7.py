class Solution(object):

    def convertToBase7(self, num):
        if False:
            while True:
                i = 10
        if num < 0:
            return '-' + self.convertToBase7(-num)
        result = ''
        while num:
            result = str(num % 7) + result
            num //= 7
        return result if result else '0'

class Solution2(object):

    def convertToBase7(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: str\n        '
        if num < 0:
            return '-' + self.convertToBase7(-num)
        if num < 7:
            return str(num)
        return self.convertToBase7(num // 7) + str(num % 7)
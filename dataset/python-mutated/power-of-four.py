class Solution(object):

    def isPowerOfFour(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: bool\n        '
        return num > 0 and num & num - 1 == 0 and (num & 1431655765 == num)

class Solution2(object):

    def isPowerOfFour(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: bool\n        '
        while num and (not num & 3):
            num >>= 2
        return num == 1

class Solution3(object):

    def isPowerOfFour(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: bool\n        '
        num = bin(num)
        return True if num[2:].startswith('1') and len(num[2:]) == num.count('0') and num.count('0') % 2 and ('-' not in num) else False
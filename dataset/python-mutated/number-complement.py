class Solution(object):

    def findComplement(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: int\n        '
        return 2 ** (len(bin(num)) - 2) - 1 - num

class Solution2(object):

    def findComplement(self, num):
        if False:
            print('Hello World!')
        i = 1
        while i <= num:
            i <<= 1
        return i - 1 ^ num

class Solution3(object):

    def findComplement(self, num):
        if False:
            for i in range(10):
                print('nop')
        bits = '{0:b}'.format(num)
        complement_bits = ''.join(('1' if bit == '0' else '0' for bit in bits))
        return int(complement_bits, 2)
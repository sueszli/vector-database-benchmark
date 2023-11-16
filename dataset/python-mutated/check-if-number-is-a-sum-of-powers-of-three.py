class Solution(object):

    def checkPowersOfThree(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: bool\n        '
        while n > 0:
            if n % 3 == 2:
                return False
            n //= 3
        return True
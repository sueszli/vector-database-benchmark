class Solution(object):

    def isUgly(self, num):
        if False:
            for i in range(10):
                print('nop')
        if num == 0:
            return False
        for i in [2, 3, 5]:
            while num % i == 0:
                num /= i
        return num == 1
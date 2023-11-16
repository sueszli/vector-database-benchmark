class Solution(object):
    """
    :type num: int
    :rtype: int
    """

    def addDigits(self, num):
        if False:
            for i in range(10):
                print('nop')
        return (num - 1) % 9 + 1 if num > 0 else 0
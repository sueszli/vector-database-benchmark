class Solution(object):

    def maximum69Number(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: int\n        '
        (curr, base, change) = (num, 3, 0)
        while curr:
            if curr % 10 == 6:
                change = base
            base *= 10
            curr //= 10
        return num + change

class Solution2(object):

    def maximum69Number(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: int\n        '
        return int(str(num).replace('6', '9', 1))
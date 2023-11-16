class Solution(object):

    def distMoney(self, money, children):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type money: int\n        :type children: int\n        :rtype: int\n        '
        if money < children * 1:
            return -1
        money -= children * 1
        (q, r) = divmod(money, 7)
        return min(q, children) - int(q > children or (q == children and r != 0) or (q == children - 1 and r == 3))

class Solution2(object):

    def distMoney(self, money, children):
        if False:
            print('Hello World!')
        '\n        :type money: int\n        :type children: int\n        :rtype: int\n        '
        if money < children * 1:
            return -1
        money -= children * 1
        (q, r) = divmod(money, 7)
        if q > children:
            return children - 1
        if q == children:
            return q - int(r != 0)
        if q == children - 1:
            return q - int(r == 3)
        return q
class Bank(object):

    def __init__(self, balance):
        if False:
            i = 10
            return i + 15
        '\n        :type balance: List[int]\n        '
        self.__balance = balance

    def transfer(self, account1, account2, money):
        if False:
            while True:
                i = 10
        '\n        :type account1: int\n        :type account2: int\n        :type money: int\n        :rtype: bool\n        '
        if 1 <= account2 <= len(self.__balance) and self.withdraw(account1, money):
            return self.deposit(account2, money)
        return False

    def deposit(self, account, money):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type account: int\n        :type money: int\n        :rtype: bool\n        '
        if 1 <= account <= len(self.__balance):
            self.__balance[account - 1] += money
            return True
        return False

    def withdraw(self, account, money):
        if False:
            print('Hello World!')
        '\n        :type account: int\n        :type money: int\n        :rtype: bool\n        '
        if 1 <= account <= len(self.__balance) and self.__balance[account - 1] >= money:
            self.__balance[account - 1] -= money
            return True
        return False
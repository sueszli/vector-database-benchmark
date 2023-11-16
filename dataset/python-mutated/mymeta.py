class mytype(type):

    @staticmethod
    def __new__(meta, name, bases, __dict__):
        if False:
            i = 10
            return i + 15
        print('Creating class :', name)
        print('Base classes   :', bases)
        print('Attributes     :', list(__dict__.keys()))
        return super().__new__(meta, name, bases, __dict__)

class myobject(metaclass=mytype):
    pass

class Stock(myobject):

    def __init__(self, name, shares, price):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.shares = shares
        self.price = price

    def cost(self):
        if False:
            print('Hello World!')
        return self.shares * self.price

    def sell(self, nshares):
        if False:
            return 10
        self.shares -= nshares

class MyStock(Stock):
    pass
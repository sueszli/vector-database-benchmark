import cython

@cython.cclass
class CheeseShop:
    cheeses: object

    def __cinit__(self):
        if False:
            print('Hello World!')
        self.cheeses = []

    @property
    def cheese(self):
        if False:
            while True:
                i = 10
        return "We don't have: %s" % self.cheeses

    @cheese.setter
    def cheese(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.cheeses.append(value)

    @cheese.deleter
    def cheese(self):
        if False:
            while True:
                i = 10
        del self.cheeses[:]
from cheesy import CheeseShop
shop = CheeseShop()
print(shop.cheese)
shop.cheese = 'camembert'
print(shop.cheese)
shop.cheese = 'cheddar'
print(shop.cheese)
del shop.cheese
print(shop.cheese)
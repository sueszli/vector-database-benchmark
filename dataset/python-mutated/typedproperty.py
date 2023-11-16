def typedproperty(name, expected_type):
    if False:
        i = 10
        return i + 15
    private_name = '_' + name

    @property
    def value(self):
        if False:
            return 10
        return getattr(self, private_name)

    @value.setter
    def value(self, val):
        if False:
            return 10
        if not isinstance(val, expected_type):
            raise TypeError(f'Expected {expected_type}')
        setattr(self, private_name, val)
    return value
String = lambda name: typedproperty(name, str)
Integer = lambda name: typedproperty(name, int)
Float = lambda name: typedproperty(name, float)
if __name__ == '__main__':

    class Stock:
        name = String('name')
        shares = Integer('shares')
        price = Float('price')

        def __init__(self, name, shares, price):
            if False:
                for i in range(10):
                    print('nop')
            self.name = name
            self.shares = shares
            self.price = price
"""
Topic: 类型检查系统
Desc : 
"""

class Descriptor:

    def __init__(self, name=None, **opts):
        if False:
            print('Hello World!')
        self.name = name
        for (key, value) in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        if False:
            return 10
        instance.__dict__[self.name] = value

class Typed(Descriptor):
    expected_type = type(None)

    def __set__(self, instance, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, self.expected_type):
            raise TypeError('expected ' + str(self.expected_type))
        super().__set__(instance, value)

class Unsigned(Descriptor):

    def __set__(self, instance, value):
        if False:
            i = 10
            return i + 15
        if value < 0:
            raise ValueError('Expected >= 0')
        super().__set__(instance, value)

class MaxSized(Descriptor):

    def __init__(self, name=None, **opts):
        if False:
            i = 10
            return i + 15
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super().__set__(instance, value)
if __name__ == '__main__':

    class Integer(Typed):
        expected_type = int

    class UnsignedInteger(Integer, Unsigned):
        pass

    class Float(Typed):
        expected_type = float

    class UnsignedFloat(Float, Unsigned):
        pass

    class String(Typed):
        expected_type = str

    class SizedString(String, MaxSized):
        pass

    class Stock:
        name = SizedString('name', size=8)
        shares = UnsignedInteger('shares')
        price = UnsignedFloat('price')

        def __init__(self, name, shares, price):
            if False:
                for i in range(10):
                    print('nop')
            self.name = name
            self.shares = shares
            self.price = price
    s = Stock('ACME', 50, 91.1)

    def check_attributes(**kwargs):
        if False:
            i = 10
            return i + 15

        def decorate(cls):
            if False:
                i = 10
                return i + 15
            for (key, value) in kwargs.items():
                if isinstance(value, Descriptor):
                    value.name = key
                    setattr(cls, key, value)
                else:
                    setattr(cls, key, value(key))
            return cls
        return decorate

    @check_attributes(name=SizedString(size=8), shares=UnsignedInteger, price=UnsignedFloat)
    class Stock:

        def __init__(self, name, shares, price):
            if False:
                for i in range(10):
                    print('nop')
            self.name = name
            self.shares = shares
            self.price = price

class checkedmeta(type):

    def __new__(cls, clsname, bases, methods):
        if False:
            i = 10
            return i + 15
        for (key, value) in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)

class Stock2(metaclass=checkedmeta):
    name = SizedString(size=8)
    shares = UnsignedInteger()
    price = UnsignedFloat()

    def __init__(self, name, shares, price):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.shares = shares
        self.price = price

class Descriptor:

    def __init__(self, name=None, **opts):
        if False:
            i = 10
            return i + 15
        self.name = name
        for (key, value) in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        if False:
            i = 10
            return i + 15
        instance.__dict__[self.name] = value

def Typed(expected_type, cls=None):
    if False:
        i = 10
        return i + 15
    if cls is None:
        return lambda cls: Typed(expected_type, cls)
    super_set = cls.__set__

    def __set__(self, instance, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, expected_type):
            raise TypeError('expected ' + str(expected_type))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

def Unsigned(cls):
    if False:
        return 10
    super_set = cls.__set__

    def __set__(self, instance, value):
        if False:
            print('Hello World!')
        if value < 0:
            raise ValueError('Expected >= 0')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

def MaxSized(cls):
    if False:
        print('Hello World!')
    super_init = cls.__init__

    def __init__(self, name=None, **opts):
        if False:
            while True:
                i = 10
        if 'size' not in opts:
            raise TypeError('missing size option')
        super_init(self, name, **opts)
    cls.__init__ = __init__
    super_set = cls.__set__

    def __set__(self, instance, value):
        if False:
            return 10
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls

@Typed(int)
class Integer(Descriptor):
    pass

@Unsigned
class UnsignedInteger(Integer):
    pass

@Typed(float)
class Float(Descriptor):
    pass

@Unsigned
class UnsignedFloat(Float):
    pass

@Typed(str)
class String(Descriptor):
    pass

@MaxSized
class SizedString(String):
    pass
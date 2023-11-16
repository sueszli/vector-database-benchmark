class Base(object):
    """
    >>> base = Base()
    >>> print(base.noargs())
    Base
    >>> print(base.int_arg(1))
    Base
    >>> print(base._class())
    Base
    """

    def noargs(self):
        if False:
            while True:
                i = 10
        return 'Base'

    def int_arg(self, i):
        if False:
            while True:
                i = 10
        return 'Base'

    @classmethod
    def _class(tp):
        if False:
            for i in range(10):
                print('nop')
        return 'Base'

class Derived(Base):
    """
    >>> derived = Derived()
    >>> print(derived.noargs())
    Derived
    >>> print(derived.int_arg(1))
    Derived
    >>> print(derived._class())
    Derived
    """

    def noargs(self):
        if False:
            return 10
        return 'Derived'

    def int_arg(self, i):
        if False:
            i = 10
            return i + 15
        return 'Derived'

    @classmethod
    def _class(tp):
        if False:
            i = 10
            return i + 15
        return 'Derived'

class DerivedDerived(Derived):
    """
    >>> derived = DerivedDerived()
    >>> print(derived.noargs())
    DerivedDerived
    >>> print(derived.int_arg(1))
    DerivedDerived
    >>> print(derived._class())
    DerivedDerived
    """

    def noargs(self):
        if False:
            while True:
                i = 10
        return 'DerivedDerived'

    def int_arg(self, i):
        if False:
            i = 10
            return i + 15
        return 'DerivedDerived'

    @classmethod
    def _class(tp):
        if False:
            for i in range(10):
                print('nop')
        return 'DerivedDerived'

class Derived2(Base):
    """
    >>> derived = Derived2()
    >>> print(derived.noargs())
    Derived2
    >>> print(derived.int_arg(1))
    Derived2
    >>> print(derived._class())
    Derived2
    """

    def noargs(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Derived2'

    def int_arg(self, i):
        if False:
            for i in range(10):
                print('nop')
        return 'Derived2'

    @classmethod
    def _class(tp):
        if False:
            for i in range(10):
                print('nop')
        return 'Derived2'
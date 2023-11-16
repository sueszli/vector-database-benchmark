class A:
    """A class having no __init__, no __new__"""

class B:
    """A class having __init__(no docstring), no __new__"""

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class C:
    """A class having __init__, no __new__"""

    def __init__(self):
        if False:
            print('Hello World!')
        '__init__ docstring'

class D:
    """A class having no __init__, __new__(no docstring)"""

    def __new__(cls):
        if False:
            return 10
        pass

class E:
    """A class having no __init__, __new__"""

    def __new__(cls):
        if False:
            i = 10
            return i + 15
        '__new__ docstring'

class F:
    """A class having both __init__ and __new__"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '__init__ docstring'

    def __new__(cls):
        if False:
            for i in range(10):
                print('nop')
        '__new__ docstring'

class G(C):
    """A class inherits __init__ without docstring."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class H(E):
    """A class inherits __new__ without docstring."""

    def __init__(self):
        if False:
            while True:
                i = 10
        pass
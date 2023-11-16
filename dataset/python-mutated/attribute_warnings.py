"""
Jedi issues warnings for possible errors if ``__getattr__``,
``__getattribute__`` or ``setattr`` are used.
"""

class Cls:

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(str, name)
Cls().upper
Cls().undefined

class Inherited(Cls):
    pass
Inherited().upper
Inherited().undefined

class SetattrCls:

    def __init__(self, dct):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in dct.items():
            setattr(self, k, v)
        self.defined = 3
c = SetattrCls({'a': 'b'})
c.defined
c.undefined
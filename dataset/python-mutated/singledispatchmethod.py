from functools import singledispatchmethod

class Foo:
    """docstring"""

    @singledispatchmethod
    def meth(self, arg, kwarg=None):
        if False:
            return 10
        'A method for general use.'
        pass

    @meth.register(int)
    @meth.register(float)
    def _meth_int(self, arg, kwarg=None):
        if False:
            for i in range(10):
                print('nop')
        'A method for int.'
        pass

    @meth.register(str)
    def _meth_str(self, arg, kwarg=None):
        if False:
            return 10
        'A method for str.'
        pass

    @meth.register
    def _meth_dict(self, arg: dict, kwarg=None):
        if False:
            while True:
                i = 10
        'A method for dict.'
        pass
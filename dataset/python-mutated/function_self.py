import cython

def regular(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> hasattr(regular, "__self__")\n    False\n    >>> nested = regular(10)\n    >>> hasattr(nested, "__self__")\n    False\n    '

    def nested(y):
        if False:
            while True:
                i = 10
        return x + y
    return nested

@cython.locals(x=cython.floating)
def fused(x):
    if False:
        return 10
    '\n    >>> nested = fused(10.)\n    >>> hasattr(nested, "__self__")\n    False\n\n    >>> hasattr(fused, "__self__")\n    False\n    '

    def nested_in_fused(y):
        if False:
            for i in range(10):
                print('nop')
        return x + y
    return nested_in_fused

class C:
    """
    >>> c = C()
    >>> c.regular.__self__ is c
    True
    >>> c.fused.__self__ is c
    True

    >>> hasattr(C.regular, "__self__")  # __self__==None on pure-python 2
    False

    >>> C.fused.__self__  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    AttributeError: ...__self__...
    """

    def regular(self):
        if False:
            return 10
        pass

    @cython.locals(x=cython.floating)
    def fused(self, x):
        if False:
            return 10
        return x
if cython.compiled:
    __doc__ = "\n    >>> hasattr(fused['double'], '__self__')\n    False\n\n    >>> hasattr(C.fused['double'], '__self__')\n    False\n\n    >>> c = C()\n    >>> c.fused['double'].__self__ is c\n    True\n\n    # The PR that changed __self__ also changed how __doc__ is set up slightly\n    >>> fused['double'].__doc__ == fused.__doc__ and isinstance(fused.__doc__, str)\n    True\n"
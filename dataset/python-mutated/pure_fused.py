import cython
InPy = cython.fused_type(cython.int, cython.float)

class TestCls:

    def func1(self, arg: 'NotInPy'):
        if False:
            for i in range(10):
                print('nop')
        "\n        >>> TestCls().func1(1.0)\n        'float'\n        >>> TestCls().func1(2)\n        'int'\n        "
        loc: 'NotInPy' = arg
        return cython.typeof(arg)
    if cython.compiled:

        @cython.locals(arg=NotInPy, loc=NotInPy)
        def func2(self, arg):
            if False:
                print('Hello World!')
            "\n            >>> TestCls().func2(1.0)\n            'float'\n            >>> TestCls().func2(2)\n            'int'\n            "
            loc = arg
            return cython.typeof(arg)

    def cpfunc(self, arg):
        if False:
            i = 10
            return i + 15
        "\n        >>> TestCls().cpfunc(1.0)\n        'float'\n        >>> TestCls().cpfunc(2)\n        'int'\n        "
        loc = arg
        return cython.typeof(arg)

    def func1_inpy(self, arg: InPy):
        if False:
            i = 10
            return i + 15
        "\n        >>> TestCls().func1_inpy(1.0)\n        'float'\n        >>> TestCls().func1_inpy(2)\n        'int'\n        "
        loc: InPy = arg
        return cython.typeof(arg)

    @cython.locals(arg=InPy, loc=InPy)
    def func2_inpy(self, arg):
        if False:
            i = 10
            return i + 15
        "\n        >>> TestCls().func2_inpy(1.0)\n        'float'\n        >>> TestCls().func2_inpy(2)\n        'int'\n        "
        loc = arg
        return cython.typeof(arg)
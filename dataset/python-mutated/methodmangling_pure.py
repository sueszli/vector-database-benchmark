import cython
if cython.compiled:

    def declare(**kwargs):
        if False:
            i = 10
            return i + 15
        return kwargs['__x']

    class RegularClass:

        @cython.locals(__x=cython.int)
        def f1(self, __x, dummy=None):
            if False:
                return 10
            '\n            Is the locals decorator correctly applied\n            >>> c = RegularClass()\n            >>> c.f1(1)\n            1\n            >>> c.f1("a")\n            Traceback (most recent call last):\n            ...\n            TypeError: an integer is required\n            >>> c.f1(_RegularClass__x = 1)\n            1\n            '
            return __x

        def f2(self, x):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Is the locals decorator correctly applied\n            >>> c = RegularClass()\n            >>> c.f2(1)\n            1\n            >>> c.f2("a")\n            Traceback (most recent call last):\n            ...\n            TypeError: an integer is required\n            '
            __x = cython.declare(cython.int, x)
            return __x

        def f3(self, x):
            if False:
                print('Hello World!')
            '\n            Is the locals decorator correctly applied\n            >>> c = RegularClass()\n            >>> c.f3(1)\n            1\n            >>> c.f3("a")\n            Traceback (most recent call last):\n            ...\n            TypeError: an integer is required\n            '
            cython.declare(__x=cython.int)
            __x = x
            return __x

        def f4(self, x):
            if False:
                i = 10
                return i + 15
            '\n            We shouldn\'t be tripped up by a function called\n            "declare" that is nothing to do with cython\n            >>> RegularClass().f4(1)\n            1\n            '
            return declare(__x=x)
else:
    __doc__ = '\n    >>> True\n    True\n    '
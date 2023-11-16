"""
Test that fused functions can be used in the same way as CyFunctions with respect to
assigning them to class attributes. Previously they enforced extra type/argument checks
beyond those which CyFunctions did.
"""
import cython
MyFusedClass = cython.fused_type(float, 'Cdef', object)

def fused_func(x: MyFusedClass):
    if False:
        print('Hello World!')
    return (type(x).__name__, cython.typeof(x))
IntOrFloat = cython.fused_type(int, float)

def fused_func_0(x: IntOrFloat=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Fused functions can legitimately take 0 arguments\n    >>> fused_func_0()\n    ('int', 'int')\n\n    # subscripted in module __doc__ conditionally\n    "
    return (type(x).__name__, cython.typeof(x))

def regular_func(x):
    if False:
        for i in range(10):
            print('nop')
    return (type(x).__name__, cython.typeof(x))

def regular_func_0():
    if False:
        while True:
            i = 10
    return

@classmethod
def fused_classmethod_free(cls, x: IntOrFloat):
    if False:
        return 10
    return (cls.__name__, type(x).__name__)

@cython.cclass
class Cdef:
    __doc__ = "\n    >>> c = Cdef()\n\n    # functions are callable with an instance of c\n    >>> c.fused_func()\n    ('Cdef', 'Cdef')\n    >>> c.regular_func()\n    ('Cdef', '{typeofCdef}')\n    >>> c.fused_in_class(1.5)\n    ('float', 'float')\n\n    # Fused functions are callable without an instance\n    # (This applies to everything in Py3 - see __doc__ below)\n    >>> Cdef.fused_func(1.5)\n    ('float', 'float')\n    >>> Cdef.fused_in_class(c, 1.5)\n    ('float', 'float')\n    >>> Cdef.fused_func_0()\n    ('int', 'int')\n\n    # Functions not expecting an argument don't work with an instance\n    >>> c.regular_func_0()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: regular_func_0() takes ... arguments ...1... given...\n\n    # Looking up a class attribute doesn't go through all of __get__\n    >>> Cdef.fused_in_class is Cdef.fused_in_class\n    True\n\n    # looking up a classmethod does go through __get__ though\n    >>> Cdef.fused_classmethod is Cdef.fused_classmethod\n    False\n    >>> Cdef.fused_classmethod_free is Cdef.fused_classmethod_free\n    False\n    >>> Cdef.fused_classmethod(1)\n    ('Cdef', 'int')\n    >>> Cdef.fused_classmethod_free(1)\n    ('Cdef', 'int')\n    ".format(typeofCdef='Python object' if cython.compiled else 'Cdef')
    if cython.compiled:
        __doc__ += '\n\n    # fused_func_0 does not accept a "Cdef" instance\n    >>> c.fused_func_0()\n    Traceback (most recent call last):\n    TypeError: No matching signature found\n\n    # subscripting requires fused methods (so  not pure Python)\n    >>> Cdef.fused_func_0[\'float\']()\n    (\'float\', \'float\')\n    >>> c.fused_func_0[\'float\']()  # doctest: +IGNORE_EXCEPTION_DETAIL\n    Traceback (most recent call last):\n    TypeError: (Exception looks quite different in Python2 and 3 so no way to match both)\n\n    >>> Cdef.fused_classmethod[\'float\'] is Cdef.fused_classmethod[\'float\']\n    False\n    >>> Cdef.fused_classmethod_free[\'float\'] is Cdef.fused_classmethod_free[\'float\']\n    False\n    '
    fused_func = fused_func
    fused_func_0 = fused_func_0
    regular_func = regular_func
    regular_func_0 = regular_func_0
    fused_classmethod_free = fused_classmethod_free

    def fused_in_class(self, x: MyFusedClass):
        if False:
            i = 10
            return i + 15
        return (type(x).__name__, cython.typeof(x))

    def regular_in_class(self):
        if False:
            for i in range(10):
                print('nop')
        return type(self).__name__

    @classmethod
    def fused_classmethod(cls, x: IntOrFloat):
        if False:
            return 10
        return (cls.__name__, type(x).__name__)

class Regular(object):
    __doc__ = "\n    >>> c = Regular()\n\n    # Functions are callable with an instance of C\n    >>> c.fused_func()\n    ('Regular', '{typeofRegular}')\n    >>> c.regular_func()\n    ('Regular', '{typeofRegular}')\n\n    # Fused functions are callable without an instance\n    # (This applies to everything in Py3 - see __doc__ below)\n    >>> Regular.fused_func(1.5)\n    ('float', 'float')\n    >>> Regular.fused_func_0()\n    ('int', 'int')\n\n    # Functions not expecting an argument don't work with an instance\n    >>> c.regular_func_0()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: regular_func_0() takes ... arguments ...1... given...\n\n    # Looking up a class attribute doesn't go through all of __get__\n    >>> Regular.fused_func is Regular.fused_func\n    True\n\n    # looking up a classmethod does go __get__ though\n    >>> Regular.fused_classmethod is Regular.fused_classmethod\n    False\n    >>> Regular.fused_classmethod_free is Regular.fused_classmethod_free\n    False\n    >>> Regular.fused_classmethod(1)\n    ('Regular', 'int')\n    >>> Regular.fused_classmethod_free(1)\n    ('Regular', 'int')\n    ".format(typeofRegular='Python object' if cython.compiled else 'Regular')
    if cython.compiled:
        __doc__ += '\n    # fused_func_0 does not accept a "Regular" instance\n    >>> c.fused_func_0()\n    Traceback (most recent call last):\n    TypeError: No matching signature found\n\n    # subscripting requires fused methods (so  not pure Python)\n    >>> c.fused_func_0[\'float\']()  # doctest: +IGNORE_EXCEPTION_DETAIL\n    Traceback (most recent call last):\n    TypeError: (Exception looks quite different in Python2 and 3 so no way to match both)\n    >>> Regular.fused_func_0[\'float\']()\n    (\'float\', \'float\')\n\n    >>> Regular.fused_classmethod[\'float\'] is Regular.fused_classmethod[\'float\']\n    False\n    >>> Regular.fused_classmethod_free[\'float\'] is Regular.fused_classmethod_free[\'float\']\n    False\n    '
    fused_func = fused_func
    fused_func_0 = fused_func_0
    regular_func = regular_func
    regular_func_0 = regular_func_0
    fused_classmethod_free = fused_classmethod_free

    @classmethod
    def fused_classmethod(cls, x: IntOrFloat):
        if False:
            i = 10
            return i + 15
        return (cls.__name__, type(x).__name__)
__doc__ = "\n    >>> Cdef.regular_func(1.5)\n    ('float', '{typeoffloat}')\n    >>> Regular.regular_func(1.5)\n    ('float', '{typeoffloat}')\n    >>> Cdef.regular_func_0()\n    >>> Regular.regular_func_0()\n".format(typeoffloat='Python object' if cython.compiled else 'float')
if cython.compiled:
    __doc__ += "\n    >>> fused_func_0['float']()\n    ('float', 'float')\n    "
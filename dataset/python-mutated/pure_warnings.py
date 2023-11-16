import cython
import typing
from cython.cimports.libc import stdint

def main():
    if False:
        for i in range(10):
            print('nop')
    foo1: typing.Tuple = None
    foo1: typing.Bar = None
    foo2: Bar = 1
    foo3: int = 1
    foo4: cython.int = 1
    foo5: stdint.bar = 5
    foo6: object = 1
    foo7: cython.bar = 1
    foo8: (1 + x).b
    foo9: mod.a.b
    foo10: func().b
    foo11: Bar[:, :, :]
    foo12: cython.int[:, ::1]
    with cython.annotation_typing(False):
        foo8: Bar = 1
        foo9: stdint.bar = 5
        foo10: cython.bar = 1

@cython.cfunc
def bar() -> cython.bar:
    if False:
        for i in range(10):
            print('nop')
    pass

@cython.cfunc
def bar2() -> Bar:
    if False:
        print('Hello World!')
    pass

@cython.cfunc
def bar3() -> stdint.bar:
    if False:
        print('Hello World!')
    pass

def bar4(a: cython.foo[:]):
    if False:
        print('Hello World!')
    pass
_WARNINGS = "\n12:10: Unknown type declaration 'Bar' in annotation, ignoring\n15:16: Unknown type declaration 'stdint.bar' in annotation, ignoring\n18:17: Unknown type declaration in annotation, ignoring\n19:15: Unknown type declaration in annotation, ignoring\n20:17: Unknown type declaration in annotation, ignoring\n21:14: Unknown type declaration in annotation, ignoring\n35:14: Unknown type declaration 'Bar' in annotation, ignoring\n39:20: Unknown type declaration 'stdint.bar' in annotation, ignoring\n\n# Spurious warnings from utility code - not part of the core test\n25:10: 'cpdef_method' redeclared\n36:10: 'cpdef_cname_method' redeclared\n961:29: Ambiguous exception value, same as default return value: 0\n961:29: Ambiguous exception value, same as default return value: 0\n1002:46: Ambiguous exception value, same as default return value: 0\n1002:46: Ambiguous exception value, same as default return value: 0\n1092:29: Ambiguous exception value, same as default return value: 0\n1092:29: Ambiguous exception value, same as default return value: 0\n"
_ERRORS = "\n17:16: Unknown type declaration 'cython.bar' in annotation\n30:13: Not a type\n30:19: Unknown type declaration 'cython.bar' in annotation\n35:14: Not a type\n39:14: Not a type\n42:18: Unknown type declaration 'cython.foo[:]' in annotation\n"
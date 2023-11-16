import cython
try:
    from typing import Optional
except ImportError:
    pass

def optional_cython_types(i: Optional[cython.int], d: Optional[cython.double], f: Optional[cython.float], c: Optional[cython.complex], l: Optional[cython.long], ll: Optional[cython.longlong]):
    if False:
        i = 10
        return i + 15
    pass
MyStruct = cython.struct(a=cython.int, b=cython.double)

def optional_cstruct(x: Optional[MyStruct]):
    if False:
        while True:
            i = 10
    pass

def optional_pytypes(i: Optional[int], f: Optional[float], c: Optional[complex], l: Optional[long]):
    if False:
        return 10
    pass

def optional_memoryview(d: double[:], o: Optional[double[:]]):
    if False:
        print('Hello World!')
    pass
_ERRORS = '\n13:44: typing.Optional[...] cannot be applied to type int\n13:69: typing.Optional[...] cannot be applied to type double\n13:97: typing.Optional[...] cannot be applied to type float\n14:44: typing.Optional[...] cannot be applied to type double complex\n14:73: typing.Optional[...] cannot be applied to type long\n14:100: typing.Optional[...] cannot be applied to type long long\n\n20:33: typing.Optional[...] cannot be applied to type MyStruct\n'
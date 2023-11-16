import os
from ctypes import c_long, sizeof
from functools import reduce
from typing import Tuple as tTuple, Type
from sympy.external import import_module
from .pythonmpq import PythonMPQ
from .ntheory import bit_scan1 as python_bit_scan1, bit_scan0 as python_bit_scan0, remove as python_remove, factorial as python_factorial, sqrt as python_sqrt, sqrtrem as python_sqrtrem, gcd as python_gcd, lcm as python_lcm, gcdext as python_gcdext, is_square as python_is_square, invert as python_invert, legendre as python_legendre, jacobi as python_jacobi, kronecker as python_kronecker, iroot as python_iroot, is_fermat_prp as python_is_fermat_prp, is_euler_prp as python_is_euler_prp, is_strong_prp as python_is_strong_prp, is_fibonacci_prp as python_is_fibonacci_prp, is_lucas_prp as python_is_lucas_prp, is_selfridge_prp as python_is_selfridge_prp, is_strong_lucas_prp as python_is_strong_lucas_prp, is_strong_selfridge_prp as python_is_strong_selfridge_prp, is_bpsw_prp as python_is_bpsw_prp, is_strong_bpsw_prp as python_is_strong_bpsw_prp
__all__ = ['GROUND_TYPES', 'HAS_GMPY', 'SYMPY_INTS', 'MPQ', 'MPZ', 'bit_scan1', 'bit_scan0', 'remove', 'factorial', 'sqrt', 'is_square', 'sqrtrem', 'gcd', 'lcm', 'gcdext', 'invert', 'legendre', 'jacobi', 'kronecker', 'iroot', 'is_fermat_prp', 'is_euler_prp', 'is_strong_prp', 'is_fibonacci_prp', 'is_lucas_prp', 'is_selfridge_prp', 'is_strong_lucas_prp', 'is_strong_selfridge_prp', 'is_bpsw_prp', 'is_strong_bpsw_prp']
GROUND_TYPES = os.environ.get('SYMPY_GROUND_TYPES', 'auto').lower()
if GROUND_TYPES in ('auto', 'gmpy', 'gmpy2'):
    gmpy = import_module('gmpy2', min_module_version='2.0.0', module_version_attr='version', module_version_attr_call_args=())
    flint = None
    if gmpy is None:
        if GROUND_TYPES != 'auto':
            from warnings import warn
            warn("gmpy library is not installed, switching to 'python' ground types")
        GROUND_TYPES = 'python'
    else:
        GROUND_TYPES = 'gmpy'
elif GROUND_TYPES == 'flint':
    flint = import_module('flint')
    gmpy = None
    if flint is None:
        from warnings import warn
        warn("python_flint is not installed, switching to 'python' ground types")
        GROUND_TYPES = 'python'
    else:
        GROUND_TYPES = 'flint'
elif GROUND_TYPES == 'python':
    gmpy = None
    flint = None
    GROUND_TYPES = 'python'
else:
    from warnings import warn
    warn("SYMPY_GROUND_TYPES environment variable unrecognised. Should be 'python', 'auto', 'gmpy', or 'gmpy2'")
    gmpy = None
    flint = None
    GROUND_TYPES = 'python'
SYMPY_INTS: tTuple[Type, ...]
LONG_MAX = (1 << 8 * sizeof(c_long) - 1) - 1
if GROUND_TYPES == 'gmpy':
    HAS_GMPY = 2
    GROUND_TYPES = 'gmpy'
    SYMPY_INTS = (int, type(gmpy.mpz(0)))
    MPZ = gmpy.mpz
    MPQ = gmpy.mpq
    bit_scan1 = gmpy.bit_scan1
    bit_scan0 = gmpy.bit_scan0
    remove = gmpy.remove
    factorial = gmpy.fac
    sqrt = gmpy.isqrt
    is_square = gmpy.is_square
    sqrtrem = gmpy.isqrt_rem
    gcd = gmpy.gcd
    lcm = gmpy.lcm
    gcdext = gmpy.gcdext
    invert = gmpy.invert
    legendre = gmpy.legendre
    jacobi = gmpy.jacobi
    kronecker = gmpy.kronecker

    def iroot(x, n):
        if False:
            return 10
        if n <= LONG_MAX:
            return gmpy.iroot(x, n)
        return python_iroot(x, n)
    is_fermat_prp = gmpy.is_fermat_prp
    is_euler_prp = gmpy.is_euler_prp
    is_strong_prp = gmpy.is_strong_prp
    is_fibonacci_prp = gmpy.is_fibonacci_prp
    is_lucas_prp = gmpy.is_lucas_prp
    is_selfridge_prp = gmpy.is_selfridge_prp
    is_strong_lucas_prp = gmpy.is_strong_lucas_prp
    is_strong_selfridge_prp = gmpy.is_strong_selfridge_prp
    is_bpsw_prp = gmpy.is_bpsw_prp
    is_strong_bpsw_prp = gmpy.is_strong_bpsw_prp
elif GROUND_TYPES == 'flint':
    HAS_GMPY = 0
    GROUND_TYPES = 'flint'
    SYMPY_INTS = (int, flint.fmpz)
    MPZ = flint.fmpz
    MPQ = flint.fmpq
    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial

    def sqrt(x):
        if False:
            for i in range(10):
                print('nop')
        return flint.fmpz(x).isqrt()

    def is_square(x):
        if False:
            return 10
        if x < 0:
            return False
        return flint.fmpz(x).sqrtrem()[1] == 0

    def sqrtrem(x):
        if False:
            for i in range(10):
                print('nop')
        return flint.fmpz(x).sqrtrem()

    def gcd(*args):
        if False:
            i = 10
            return i + 15
        return reduce(flint.fmpz.gcd, args, flint.fmpz(0))

    def lcm(*args):
        if False:
            return 10
        return reduce(flint.fmpz.lcm, args, flint.fmpz(1))
    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre

    def jacobi(x, y):
        if False:
            i = 10
            return i + 15
        if y <= 0 or not y % 2:
            raise ValueError('y should be an odd positive integer')
        return flint.fmpz(x).jacobi(y)
    kronecker = python_kronecker

    def iroot(x, n):
        if False:
            i = 10
            return i + 15
        if n <= LONG_MAX:
            y = flint.fmpz(x).root(n)
            return (y, y ** n == x)
        return python_iroot(x, n)
    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp
elif GROUND_TYPES == 'python':
    HAS_GMPY = 0
    GROUND_TYPES = 'python'
    SYMPY_INTS = (int,)
    MPZ = int
    MPQ = PythonMPQ
    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial
    sqrt = python_sqrt
    is_square = python_is_square
    sqrtrem = python_sqrtrem
    gcd = python_gcd
    lcm = python_lcm
    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre
    jacobi = python_jacobi
    kronecker = python_kronecker
    iroot = python_iroot
    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp
else:
    assert False